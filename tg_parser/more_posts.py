import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

api_id = 22365566
api_hash = '223d3fd1cd0fe17b38b9f93d8134a282'
session_name = 'multi_channel_session'
#phone:+7(905)502-45-17

# Каналы для парсинга: {юзернейм: (лимит, часы_назад)}
channels_config = {
    '@bbbreaking': (10000, 336),
    '@mash': (10000, 336),
    '@interfaxonline': (10000, 336),
    '@tass_agency': (10000, 336),
    '@vedomosti': (10000, 336),
    '@rian_ru': (10000, 336),
    '@gazetaru': (10000, 336),
    '@izvestia': (10000, 336),
    '@gosuslugi': (10000, 336),
    '@bbcrussian': (10000, 336),
    '@kommersant': (10000, 336),
    '@rt_russian': (10000, 336),

    '@moscowach': (10000, 336),
    '@spb_smi': (10000, 336),
    '@ngs_news': (10000, 336),
    '@kazan': (10000, 336),
    '@podslushano_chat52': (10000, 336),
    '@news_74ru': (10000, 336),
    '@yug_24_ru': (10000, 336),
    '@samara_smi': (10000, 336),

    '@alfabank': (10000, 336),
    '@alfa_investments': (10000, 336),
    '@sberbank': (10000, 336),
    '@SberInvestments': (10000, 336),
    '@bankvtb': (10000, 336),
    '@centralbank_russia': (10000, 336),
    '@tbank': (10000, 336),
    '@gazprombank': (10000, 336),

    '@russianmacro': (10000, 336),
    '@ecworld': (10000, 336),
    '@ecworldtech': (10000, 336),
    '@economika': (10000, 336),
    '@visual_capitalist_rus': (10000, 336),
    '@Econsonline': (10000, 336),

    '@MoscowExchangeOfficial': (10000, 336),
    '@cbonds': (10000, 336),
    '@dohod': (10000, 336),
    '@Bonds_lab': (10000, 336),
    '@russianjunkbonds': (10000, 336),

    '@smartlabnews': (10000, 336),
    '@investfundsru': (10000, 336),
    '@CFA_RF': (10000, 336),
    '@tb_invest_official': (10000, 336),
    '@investnique': (10000, 336),
    '@minec_russia': (10000, 336),
    '@banksta': (10000, 336),
    '@ex_fin': (10000, 336),

    '@bitkogan': (10000, 336),
    '@d_code': (10000, 336),

    '@rusipoteka': (10000, 336),
    '@ipotekahouse': (10000, 336),
    '@regcik': (10000, 336),
    '@ipotekacenter': (10000, 336),
    '@cian_official': (10000, 336),

    '@sovcomrates_msk': (10000, 336),
    '@sberometer_kurs': (10000, 336),

    '@steamrub': (10000, 336),
}


output_dir = Path(f"telegram_data_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
output_dir.mkdir(exist_ok=True)
print(f'Папка для сохранения: {output_dir}\n')


# асинхронный парсинг
async def parse_channel(client, channel, limit, hours_back):
    min_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    messages = []
    offset_id = 0

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


# основная функция
async def main():
    async with TelegramClient(session_name, api_id, api_hash) as client:
        if not await client.is_user_authorized():
            print('Требуется авторизация. Введите код из Telegram...')
            await client.start()
            print('Сессия сохранена\n')

        tasks = [
            parse_channel(client, ch, lim, hours)
            for ch, (lim, hours) in channels_config.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total = 0
        for (channel, _), res in zip(channels_config.items(), results):
            if isinstance(res, Exception):
                print(f'Ошибка при парсинге {channel}: {res}')
                continue

            clean_name = channel.lstrip('@').replace('/', '_')
            filepath = output_dir / f"{clean_name}.json"

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            total += len(res)
            print(f'{channel}: {len(res)} постов в {filepath.name}')

        print(f'\n Итого собрано: {total} постов из {len(channels_config)} каналов в папке {output_dir}')


if __name__ == '__main__':
    asyncio.run(main())