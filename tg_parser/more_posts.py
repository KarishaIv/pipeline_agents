import argparse
import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing env var: {name}")
    return val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse multiple Telegram channels to JSON files")
    p.add_argument(
        "--day",
        default=None,
        help="День YYYY-MM-DD (по умолчанию: вчера).",
    )
    p.add_argument(
        "--mode",
        choices=["day", "hours"],
        default="day",
        help='Режим: "day" или "hours"',
    )
    p.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help='Сколько часов назад (для --mode=hours)',
    )
    p.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Сколько дней назад (для --mode=hours)",
    )
    p.add_argument("--limit", type=int, default=10000, help="Макс. постов на канал")
    p.add_argument(
        "--output-prefix",
        default="telegram_data",
        help="Префикс папки вывода",
    )
    p.add_argument(
        "--session",
        default=os.getenv("TELEGRAM_SESSION", "multi_channel_session"),
        help="Имя сессии Telethon",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Параллельных каналов",
    )
    p.add_argument(
        "--batch-sleep",
        type=float,
        default=1.5,
        help="Пауза между страницами (сек)",
    )
    return p.parse_args()


CHANNELS = [
    "@bbbreaking",
    "@mash",
    "@interfaxonline",
    "@tass_agency",
    "@vedomosti",
    "@rian_ru",
    "@gazetaru",
    "@izvestia",
    "@gosuslugi",
    "@bbcrussian",
    "@kommersant",
    "@rt_russian",
    "@moscowach",
    "@spb_smi",
    "@ngs_news",
    "@kazan",
    "@podslushano_chat52",
    "@news_74ru",
    "@yug_24_ru",
    "@samara_smi",
    "@alfabank",
    "@alfa_investments",
    "@sberbank",
    "@SberInvestments",
    "@bankvtb",
    "@centralbank_russia",
    "@tbank",
    "@gazprombank",
    "@russianmacro",
    "@economika",
    "@visual_capitalist_rus",
    "@Econsonline",
    "@MoscowExchangeOfficial",
    "@cbonds",
    "@dohod",
    "@Bonds_lab",
    "@russianjunkbonds",
    "@smartlabnews",
    "@investfundsru",
    "@CFA_RF",
    "@tb_invest_official",
    "@investnique",
    "@minec_russia",
    "@banksta",
    "@ex_fin",
    "@bitkogan",
    "@d_code",
    "@rusipoteka",
    "@ipotekahouse",
    "@regcik",
    "@ipotekacenter",
    "@cian_official",
    "@sovcomrates_msk",
    "@sberometer_kurs",
    "@steamrub",
]


ARGS = _parse_args()

api_id = int(_require_env("TELEGRAM_API_ID"))
api_hash = _require_env("TELEGRAM_API_HASH")
session_name = ARGS.session

if ARGS.mode == "day":
    out_suffix = "last_day"
else:
    hours_back = int(ARGS.days_back * 24) if ARGS.days_back is not None else int(ARGS.hours_back)
    out_suffix = f"last_{int((hours_back + 23) // 24)}_days"

output_dir = Path(f"{ARGS.output_prefix}_{out_suffix}")
output_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.info("Папка для сохранения: %s", output_dir)
logging.info("max_concurrent=%s batch_sleep=%ss", ARGS.max_concurrent, ARGS.batch_sleep)

_semaphore = asyncio.Semaphore(ARGS.max_concurrent)


def _day_window_utc(target_day: date) -> tuple[datetime, datetime]:
    """
    Возвращает (start_utc, end_utc) для локальных суток target_day.
    Telethon отдаёт msg.date как timezone-aware UTC.
    """
    local_tz = datetime.now().astimezone().tzinfo
    start_local = datetime.combine(target_day, datetime.min.time(), tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _resolve_target_day() -> date:
    local_today = datetime.now().astimezone().date()
    if ARGS.day:
        return date.fromisoformat(ARGS.day)
    return local_today - timedelta(days=1)


async def parse_channel(
    client: TelegramClient,
    channel: str,
    limit: int,
    hours_back: int,
    *,
    mode: str,
    day_start_utc: datetime | None,
    day_end_utc: datetime | None,
) -> list:
    async with _semaphore:
        min_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        messages: list = []
        offset_id = 0

        if mode == "day":
            assert day_start_utc is not None and day_end_utc is not None
            logging.info("Начало сбора %s (day UTC %s..%s)", channel, day_start_utc, day_end_utc)
        else:
            logging.info("Начало сбора %s (hours_back=%s)", channel, hours_back)

        try:
            while len(messages) < limit:
                try:
                    page = min(100, limit - len(messages))
                    batch = await client.get_messages(
                        channel,
                        limit=page,
                        offset_id=offset_id,
                        max_id=0,
                    )
                except FloodWaitError as e:
                    wait = min(e.seconds, 300)
                    logging.warning("FloodWait на %s: %ss", channel, wait)
                    await asyncio.sleep(wait)
                    continue

                if not batch:
                    break

                for msg in batch:
                    if not msg.date:
                        continue

                    if mode == "day":
                        if msg.date >= day_end_utc:
                            continue
                        if msg.date < day_start_utc:
                            return messages
                    else:
                        if msg.date < min_date:
                            return messages

                    messages.append(
                        {
                            "id": msg.id,
                            "date": str(msg.date),
                            "text": getattr(msg, "message", "") or "",
                            "views": getattr(msg, "views", None),
                            "channel": channel,
                        }
                    )

                if len(batch) < page:
                    break

                offset_id = batch[-1].id
                await asyncio.sleep(ARGS.batch_sleep)

            logging.info("%s: собрано %s постов", channel, len(messages))
            return messages

        except ChannelPrivateError:
            logging.error("%s: приватный канал или доступ закрыт", channel)
            return []
        except Exception as e:
            logging.error("Ошибка при парсинге %s: %s", channel, e)
            return []


async def main() -> None:
    async with TelegramClient(session_name, api_id, api_hash) as client:
        if not await client.is_user_authorized():
            logging.info("Требуется авторизация. Введите код из Telegram...")
            await client.start()
            logging.info("Сессия сохранена")

        day_start_utc = day_end_utc = None
        effective_hours_back = int(ARGS.days_back * 24) if ARGS.days_back is not None else int(ARGS.hours_back)
        if ARGS.mode == "day":
            target_day = _resolve_target_day()
            day_start_utc, day_end_utc = _day_window_utc(target_day)

        tasks = [
            parse_channel(
                client,
                ch,
                ARGS.limit,
                effective_hours_back,
                mode=ARGS.mode,
                day_start_utc=day_start_utc,
                day_end_utc=day_end_utc,
            )
            for ch in CHANNELS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total = 0
        for channel, res in zip(CHANNELS, results):
            if isinstance(res, Exception):
                logging.error("Критическая ошибка %s: %s", channel, res)
                continue

            clean_name = channel.lstrip("@").replace("/", "_")
            filepath = output_dir / f"{clean_name}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            total += len(res)
            logging.info("%s: %s постов → %s", channel, len(res), filepath.name)

        logging.info("Итого: %s постов из %s каналов в %s", total, len(CHANNELS), output_dir)


if __name__ == "__main__":
    asyncio.run(main())
