import argparse
import asyncio
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing env var: {name}")
    return val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Парсинг нескольких Telegram-каналов в JSON")
    p.add_argument(
        "--day",
        default=None,
        help="День YYYY-MM-DD; по умолчанию — вчера",
    )
    p.add_argument(
        "--mode",
        choices=["day", "hours"],
        default="day",
        help='Режим: day или hours',
    )
    p.add_argument(
        "--hours-back",
        type=int,
        default=24,
        help="Сколько часов назад при --mode=hours (по умолчанию 24)",
    )
    p.add_argument("--limit", type=int, default=10000, help="Макс. постов на канал (по умолчанию 10000)")
    p.add_argument(
        "--output-prefix",
        default="telegram_data",
        help="Префикс папки вывода (по умолчанию telegram_data)",
    )
    p.add_argument(
        "--session",
        default=os.getenv("TELEGRAM_SESSION", "multi_channel_session"),
        help="Имя сессии Telethon (env TELEGRAM_SESSION, иначе multi_channel_session)",
    )
    return p.parse_args()

# Каналы для парсинга
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
    "@ecworld",
    "@ecworldtech",
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

output_dir = Path(f"{ARGS.output_prefix}_last_day")
output_dir.mkdir(exist_ok=True)
print(f"Папка для сохранения: {output_dir}\n")


def _day_window_utc(target_day: date) -> tuple[datetime, datetime]:
    """
    Возвращает (start_utc, end_utc) для локальных суток target_day
    Telethon отдаёт msg.date как timezone-aware UTC
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


async def parse_channel(client, channel, limit, hours_back, *, mode: str, day_start_utc: datetime | None, day_end_utc: datetime | None):
    min_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    messages = []
    offset_id = 0

    if mode == "day":
        assert day_start_utc and day_end_utc
        print(f"Сбор {channel} (посты за день, UTC окно {day_start_utc.isoformat()}..{day_end_utc.isoformat()})")
    else:
        print(f'Сбор {channel} (посты за {hours_back}ч)')

    while len(messages) < limit:
        batch = await client(GetHistoryRequest(
            peer=channel,
            limit=100,
            offset_date=None,
            offset_id=offset_id,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0
        ))

        if not batch.messages:
            break

        for msg in batch.messages:
            if mode == "day":
                if msg.date >= day_end_utc: 
                    continue
                if msg.date < day_start_utc:  
                    return messages
            else:
                if msg.date < min_date:
                    print(f'{channel}: достигнута дата {min_date.strftime("%Y-%m-%d")}')
                    return messages

            messages.append({
                'id': msg.id,
                'date': str(msg.date),
                'text': getattr(msg, 'message', '') or '',
                'views': getattr(msg, 'views', None),
                'channel': channel
            })
            offset_id = msg.id

        if len(batch.messages) < 100:
            break

    return messages


async def main():
    async with TelegramClient(session_name, api_id, api_hash) as client:
        if not await client.is_user_authorized():
            print('Требуется авторизация. Введите код из Telegram...')
            await client.start()
            print('Сессия сохранена\n')

        day_start_utc = day_end_utc = None
        if ARGS.mode == "day":
            target_day = _resolve_target_day()
            day_start_utc, day_end_utc = _day_window_utc(target_day)

        tasks = [
            parse_channel(
                client,
                ch,
                ARGS.limit,
                ARGS.hours_back,
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
                print(f'Ошибка при парсинге {channel}: {res}')
                continue

            clean_name = channel.lstrip('@').replace('/', '_')
            filepath = output_dir / f"{clean_name}.json"

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            total += len(res)
            print(f'{channel}: {len(res)} постов в {filepath.name}')

        print(f'\n Итого собрано: {total} постов из {len(CHANNELS)} каналов в папке {output_dir}')


if __name__ == '__main__':
    asyncio.run(main())