import os
import sys
import django

# Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from django.conf import settings
from decimal import Decimal
from datetime import datetime, timedelta
from django.db.models import Sum, Count
from django.utils import timezone
from asgiref.sync import sync_to_async

from bot.models import TelegramUser, Conversation, BotAdmin
from rag.vectordb import get_context, search


# OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# System prompt
SYSTEM_PROMPT_LATIN = """Sen Ekologik ekspertiza markazi haqida ma'lumot beruvchi rasmiy yordamchi botsan.

SENING VAZIFANG:
Foydalanuvchi savollariga FAQAT quyidagi KONTEKST asosida to'liq, aniq va professional javob berish.

JAVOB BERISH QOIDALARI:
1. FAQAT kontekstda bor ma'lumotlardan foydalaning
2. Kontekstdagi BARCHA tegishli ma'lumotlarni to'liq sanab o'ting
3. Ro'yxatlar, tartiblar, hujjatlar bo'lsa - HECH NARSA QOLDIRMAY barchasini yozing
4. Raqamlar, sanalar, summalar, telefon raqamlarini AYNAN ko'rsating
5. Javobingizni FAQAT LOTIN alifbosida yozing (a-z harflari)

JAVOB TOPILMAGANDA:
- Agar savol ekologiya, atrof-muhit, ekspertiza mavzusiga UMUMAN TEGISHLI BO'LMASA → "MAVZU_TASHQARI"
- Agar savol mavzuga aloqador lekin kontekstda aniq javob YO'Q BO'LSA → "JAVOB_TOPILMADI"

MUHIM:
- Hech qachon o'ylab topib javob BERMANG
- Kontekstdan tashqariga CHIQMANG
- Shunchaki kontekstdagi ma'lumotni ANIQ va TO'LIQ yetkazing

KONTEKST (Rasmiy hujjatlardan):
{context}
"""

SYSTEM_PROMPT_CYRILLIC = """Сен Экологик экспертиза маркази ҳақида маълумот берувчи расмий ёрдамчи ботсан.

СЕНИНГ ВАЗИФАНГ:
Фойдаланувчи саволларига ФАҚАТ қуйидаги КОНТЕКСТ асосида тўлиқ, аниқ ва профессионал жавоб бериш.

ЖАВОБ БЕРИШ ҚОИДАЛАРИ:
1. ФАҚАТ контекстда бор маълумотлардан фойдаланинг
2. Контекстдаги БАРЧА тегишли маълумотларни тўлиқ санаб ўтинг
3. Рўйхатлар, тартиблар, ҳужжатлар бўлса - ҲЕЧ НАРСА ҚОЛДИРМАЙ барчасини ёзинг
4. Рақамлар, саналар, суммалар, телефон рақамларини АЙНАН кўрсатинг
5. Жавобингизни ФАҚАТ КИРИЛЛ алифбосида ёзинг

ЖАВОБ ТОПИЛМАГАНДА:
- Агар савол экология, атроф-муҳит, экспертиза мавзусига УМУМАН ТЕГИШЛИ БЎЛМАСА → "МАВЗУ_ТАШҚАРИ"
- Агар савол мавзуга алоқадор лекин контекстда аниқ жавоб ЙЎҚ БЎЛСА → "ЖАВОБ_ТОПИЛМАДИ"

МУҲИМ:
- Ҳеч қачон ўйлаб топиб жавоб БЕРМАНГ
- Контекстдан ташқарига ЧИҚМАНГ
- Шунчаки контекстдаги маълумотни АНИҚ ва ТЎЛИҚ етказинг

КОНТЕКСТ (Расмий ҳужжатлардан):
{context}
"""

SYSTEM_PROMPT_RUSSIAN = """Ты официальный бот-помощник Центра государственной экологической экспертизы, предоставляющий информацию.

ТВОЯ ЗАДАЧА:
Отвечать на вопросы пользователей ТОЛЬКО на основе приведённого ниже КОНТЕКСТА — полно, точно и профессионально.

ПРАВИЛА ОТВЕТА:
1. Используйте ТОЛЬКО информацию из контекста
2. Перечислите ВСЮ соответствующую информацию из контекста полностью
3. Если есть списки, процедуры, документы — напишите ВСЁ БЕЗ ПРОПУСКОВ
4. Указывайте ТОЧНЫЕ цифры, даты, суммы, номера телефонов
5. Пишите ответ ТОЛЬКО на РУССКОМ языке

КОГДА ОТВЕТ НЕ НАЙДЕН:
- Если вопрос ВООБЩЕ НЕ ОТНОСИТСЯ к теме экологии, окружающей среды, экспертизы → "MAVZU_TASHQARI"
- Если вопрос по теме, но в контексте НЕТ точного ответа → "JAVOB_TOPILMADI"

ВАЖНО:
- Никогда НЕ ВЫДУМЫВАЙТЕ ответы
- НЕ ВЫХОДИТЕ за рамки контекста
- Просто ТОЧНО и ПОЛНО передайте информацию из контекста

КОНТЕКСТ (Из официальных документов):
{context}
"""


def detect_alphabet(text: str) -> str:
    """Matnning alifbosini aniqlash: 'latin', 'cyrillic' (o'zbek), 'russian'"""
    cyrillic_count = 0
    latin_count = 0
    uzbek_specific = 0

    # O'zbek kirilliga xos harflar
    uzbek_chars = set('ўқғҳЎҚҒҲ')

    for char in text:
        if char in uzbek_chars:
            uzbek_specific += 1
            cyrillic_count += 1
        elif '\u0400' <= char <= '\u04FF':  # Cyrillic range
            cyrillic_count += 1
        elif 'a' <= char.lower() <= 'z':  # Latin range
            latin_count += 1

    if cyrillic_count > latin_count:
        # O'zbek kirilimi yoki rusmi?
        return 'cyrillic' if uzbek_specific > 0 else 'russian'
    return 'latin'

OFF_TOPIC_MESSAGE_LATIN = """Kechirasiz, men faqat O'zbekiston Respublikasi Vazirlar Mahkamasining 2020 yil 7 sentabrdagi 541-son qarori doirasida ma'lumot bera olaman.

Iltimos, savolingizni shu qaror mazmuniga oid qilib bering."""

OFF_TOPIC_MESSAGE_CYRILLIC = """Кечирасиз, мен фақат Ўзбекистон Республикаси Вазирлар Маҳкамасининг 2020 йил 7 сентябрдаги 541-сон қарори доирасида маълумот бера оламан.

Илтимос, саволингизни шу қарор мазмунига оид қилиб беринг."""

OFF_TOPIC_MESSAGE_RUSSIAN = """Извините, я могу предоставлять информацию только в рамках Постановления Кабинета Министров Республики Узбекистан №541 от 7 сентября 2020 года.

Пожалуйста, задайте вопрос по содержанию данного постановления."""

NOT_FOUND_MESSAGE_LATIN = """Kechirasiz, ushbu savol Vazirlar Mahkamasining
2020 yil 7 sentabrdagi 541-son qarori doirasiga kirmaydi.

Mazkur masala bo'yicha to'liq va aniq ma'lumot olish uchun
Davlat ekologik ekspertizasi markazi mutaxassislariga
bevosita murojaat qilishingiz mumkin:

📞 Qisqa raqam: 1392
☎️ Telefon: 71 203 03 04

Mutaxassislar sizga to'liq ma'lumot va tushuntirish beradilar."""

NOT_FOUND_MESSAGE_CYRILLIC = """Кечирасиз, ушбу савол Вазирлар Маҳкамасининг
2020 йил 7 сентябрдаги 541-сон қарори доирасига кирмайди.

Мазкур масала бўйича тўлиқ ва аниқ маълумот олиш учун
Давлат экологик экспертизаси маркази мутахассисларига
бевосита мурожаат қилишингиз мумкин:

📞 Қисқа рақам: 1392
☎️ Телефон: 71 203 03 04

Мутахассислар сизга тўлиқ маълумот ва тушунтириш берадилар."""

NOT_FOUND_MESSAGE_RUSSIAN = """Извините, данный вопрос не входит в рамки Постановления
Кабинета Министров №541 от 7 сентября 2020 года.

Для получения полной и точной информации по данному вопросу
вы можете обратиться напрямую к специалистам
Центра государственной экологической экспертизы:

📞 Короткий номер: 1392
☎️ Телефон: 71 203 03 04

Специалисты предоставят вам полную информацию и разъяснения."""


@sync_to_async
def get_or_create_user(telegram_user) -> TelegramUser:
    """Telegram user ni olish yoki yaratish"""
    user, created = TelegramUser.objects.get_or_create(
        telegram_id=telegram_user.id,
        defaults={
            'username': telegram_user.username,
            'first_name': telegram_user.first_name,
            'last_name': telegram_user.last_name
        }
    )
    return user


@sync_to_async
def save_conversation(user, question, answer, input_tokens, output_tokens, total_tokens, cost, status, source_chunks):
    """Conversation ni saqlash"""
    return Conversation.objects.create(
        user=user,
        question=question,
        answer=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
        status=status,
        source_chunks=source_chunks
    )


@sync_to_async
def check_is_admin(telegram_id: int) -> bool:
    """Admin ekanligini tekshirish"""
    return BotAdmin.objects.filter(telegram_id=telegram_id, is_active=True).exists()


@sync_to_async
def get_total_stats():
    """Umumiy statistika olish"""
    total_users = TelegramUser.objects.count()
    total_conversations = Conversation.objects.count()
    answered = Conversation.objects.filter(status='answered').count()
    not_found = Conversation.objects.filter(status='not_found').count()
    stats_data = Conversation.objects.aggregate(
        total_tokens=Sum('total_tokens'),
        total_cost=Sum('cost')
    )
    return total_users, total_conversations, answered, not_found, stats_data


@sync_to_async
def get_today_stats():
    """Bugungi statistika olish"""
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_convs = Conversation.objects.filter(created_at__gte=today_start)
    total = today_convs.count()
    answered = today_convs.filter(status='answered').count()
    not_found = today_convs.filter(status='not_found').count()
    stats_data = today_convs.aggregate(
        total_tokens=Sum('total_tokens'),
        total_cost=Sum('cost')
    )
    return total, answered, not_found, stats_data


@sync_to_async
def get_unanswered_convs():
    """Javob berilmagan savollar"""
    return list(Conversation.objects.filter(status='not_found').order_by('-created_at')[:10].select_related('user'))


@sync_to_async
def get_costs_stats():
    """Xarajatlar statistikasi"""
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_cost = Conversation.objects.filter(created_at__gte=today_start).aggregate(cost=Sum('cost'))['cost'] or 0

    week_start = today_start - timedelta(days=today_start.weekday())
    week_cost = Conversation.objects.filter(created_at__gte=week_start).aggregate(cost=Sum('cost'))['cost'] or 0

    month_start = today_start.replace(day=1)
    month_cost = Conversation.objects.filter(created_at__gte=month_start).aggregate(cost=Sum('cost'))['cost'] or 0

    total_cost = Conversation.objects.aggregate(cost=Sum('cost'))['cost'] or 0

    return today_cost, week_cost, month_cost, total_cost


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start buyrug'i"""
    # User ni saqlash
    await get_or_create_user(update.effective_user)

    await update.message.reply_text(
        "Ассалому алайкум! \n\n"
        "Мен Давлат экологик экспертизаси марказининг\n"
        "сунъий интеллектга асосланган ахборот ассистентиман.\n\n"
        "Мен сизга Ўзбекистон Республикаси Вазирлар Маҳкамасининг\n"
        "2020 йил 7 сентябрдаги 541-сон қарори\n"
        "доирасида маълумот ва тушунтиришлар бераман.\n\n"
        "✍️ Саволингизни 541-сон қарор мазмуни бўйича ёзинг.\n\n"
        "⚠️ Агар саволингиз бошқа мавзуда бўлса,\n"
        "илтимос, Марказнинг қисқа рақамига мурожаат қилинг:\n"
        "📞 1392"
    )


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi savoliga javob berish"""
    user_message = update.message.text
    user = await get_or_create_user(update.effective_user)

    # Alifboni aniqlash
    alphabet = detect_alphabet(user_message)

    # Kutish xabarini tilga qarab yuborish
    waiting_messages = {
        'latin': "⏳ Iltimos kuting, javob tayyorlanmoqda...",
        'cyrillic': "⏳ Илтимос кутинг, жавоб тайёрланмоқда...",
        'russian': "⏳ Пожалуйста, подождите, ответ готовится...",
    }
    waiting_message = await update.message.reply_text(waiting_messages[alphabet])

    try:

        # Salomlashuvlarni aniqlash
        greetings_latin = ["salom", "assalom", "hayrli kun", "xayrli kun", "hello"]
        greetings_cyrillic = ["салом", "ассалом", "хайрли кун"]
        greetings_russian = ["привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "здравствуй"]

        user_lower = user_message.lower().strip()
        is_greeting = any(q in user_lower for q in (greetings_cyrillic + greetings_latin + greetings_russian))

        # Bot haqida savollarni to'g'ridan-to'g'ri qayta ishlash
        bot_questions_cyrillic = ["сен кимсан", "бот ҳақида", "нимага ёрдам", "нима қила оласан", "сиз кимсиз", "нима билесиз"]
        bot_questions_latin = ["sen kimsan", "bot haqida", "nimaga yordam", "nima qila olasan", "siz kimsiz", "nima bilasiz"]
        bot_questions_russian = ["кто ты", "что ты умеешь", "что ты можешь", "чем помочь", "о боте", "что за бот"]

        is_bot_question = any(q in user_lower for q in (bot_questions_cyrillic + bot_questions_latin + bot_questions_russian))

        if is_greeting or is_bot_question:
            # Salomlashuvga javob + bot taqdimoti
            if alphabet == 'russian':
                greeting_text = "Здравствуйте! 😊\n\n" if is_greeting else ""
                bot_answer = f"""{greeting_text}Я официальный бот Центра государственной экологической экспертизы.

Я могу предоставить вам информацию по следующим темам:
✅ Полномочия и задачи Центра
✅ Содержание и требования Постановления №541
✅ Процесс экологической экспертизы
✅ Перечень необходимых документов
✅ Сроки и оплата
✅ Контактная информация

Задавайте ваш вопрос! 😊"""
            elif alphabet == 'cyrillic':
                greeting_text = "Ваалайкум ассалом! 😊\n\n" if is_greeting else ""
                bot_answer = f"""{greeting_text}Мен Давлат экологик экспертизаси марказининг расмий ботиман.

Мен сизга қуйидаги мавзулар бўйича маълумот бера оламан:
✅ Марказнинг ваколатлари ва вазифалари
✅ 541-сон қарор мазмуни ва талаблари
✅ Экологик экспертиза жараёни
✅ Керакли ҳужжатлар рўйхати
✅ Муддатлар ва тўловлар
✅ Алоқа маълумотлари

Саволингизни беринг! 😊"""
            else:
                greeting_text = "Vaalaykum assalom! 😊\n\n" if is_greeting else ""
                bot_answer = f"""{greeting_text}Men Davlat ekologik ekspertizasi markazining rasmiy botiman.

Men sizga quyidagi mavzular bo'yicha ma'lumot bera olaman:
✅ Markazning vakolatlari va vazifalari
✅ 541-son qaror mazmuni va talablari
✅ Ekologik ekspertiza jarayoni
✅ Kerakli hujjatlar ro'yxati
✅ Muddatlar va to'lovlar
✅ Aloqa ma'lumotlari

Savolingizni bering! 😊"""

            # Kutish xabarini o'chirish
            await waiting_message.delete()
            await update.message.reply_text(bot_answer)

            # DB ga saqlash
            await save_conversation(
                user=user,
                question=user_message,
                answer=bot_answer,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost=Decimal('0'),
                status='answered',
                source_chunks="Salomlashuv yoki bot haqida savol - to'g'ridan-to'g'ri javob"
            )
            return

        # RAG dan kontekst olish (alifboga qarab tilni tanlash)
        rag_lang = "ru" if alphabet == "russian" else "uz"
        rag_context = get_context(user_message, n_results=10, lang=rag_lang)
        source_chunks = rag_context if rag_context else "Kontekst topilmadi"

        # System prompt tayyorlash (alifboga qarab)
        if alphabet == 'russian':
            system_prompt = SYSTEM_PROMPT_RUSSIAN.format(context=rag_context if rag_context else "Информация не найдена")
        elif alphabet == 'cyrillic':
            system_prompt = SYSTEM_PROMPT_CYRILLIC.format(context=rag_context if rag_context else "Маълумот топилмади")
        else:
            system_prompt = SYSTEM_PROMPT_LATIN.format(context=rag_context if rag_context else "Ma'lumot topilmadi")

        # GPT ga yuborish (yaxshilangan model va parametrlar)
        response = client.chat.completions.create(
            model="gpt-4o",      # Eng aniq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,     # Minimal randomness - maksimal aniqlik
            max_tokens=4000,     # Ko'proq joy javob uchun
            top_p=0.95,          # Eng yuqori ehtimollik
            frequency_penalty=0.2,  # Takrorlanishni kamaytirish
            presence_penalty=0.0    # Faqat kontekstga asoslangan javob
        )

        answer = response.choices[0].message.content

        # Token va narx hisoblash (GPT-4o narxlari)
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        # GPT-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
        cost = Decimal(str((input_tokens * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)))

        # Status aniqlash
        off_topic_messages = {
            'russian': OFF_TOPIC_MESSAGE_RUSSIAN,
            'cyrillic': OFF_TOPIC_MESSAGE_CYRILLIC,
            'latin': OFF_TOPIC_MESSAGE_LATIN,
        }
        not_found_messages = {
            'russian': NOT_FOUND_MESSAGE_RUSSIAN,
            'cyrillic': NOT_FOUND_MESSAGE_CYRILLIC,
            'latin': NOT_FOUND_MESSAGE_LATIN,
        }

        if "MAVZU_TASHQARI" in answer or "МАВЗУ_ТАШҚАРИ" in answer:
            status = 'not_found'
            answer = off_topic_messages[alphabet]
        elif "JAVOB_TOPILMADI" in answer or "ЖАВОБ_ТОПИЛМАДИ" in answer:
            status = 'not_found'
            answer = not_found_messages[alphabet]
        else:
            status = 'answered'

        # DB ga saqlash
        await save_conversation(
            user=user,
            question=user_message,
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            status=status,
            source_chunks=source_chunks[:1000]
        )

        print(f"User: {user.telegram_id}, Tokens: {total_tokens}, Cost: ${cost:.6f}, Status: {status}")

        # Kutish xabarini o'chirish
        await waiting_message.delete()

        await update.message.reply_text(answer)

    except Exception as e:
        print(f"Xatolik: {e}")

        # Kutish xabarini o'chirish
        try:
            await waiting_message.delete()
        except:
            pass

        # Xatolikni ham saqlash
        await save_conversation(
            user=user,
            question=user_message,
            answer=f"Xatolik: {str(e)}",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost=Decimal('0'),
            status='not_found',
            source_chunks=""
        )

        await update.message.reply_text(
            "Kechirasiz, texnik xatolik yuz berdi. Iltimos, keyinroq urinib ko'ring yoki mutaxassis bilan bog'laning:\n"
            "+998999999999"
        )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Umumiy statistika"""
    if not await check_is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat adminlar uchun!")
        return

    total_users, total_conversations, answered, not_found, stats_data = await get_total_stats()

    message = f"""📊 Umumiy statistika:

👥 Foydalanuvchilar: {total_users}
💬 Jami savollar: {total_conversations}
✅ Javob berilgan: {answered}
❌ Javob topilmagan: {not_found}

🔢 Jami tokenlar: {stats_data['total_tokens'] or 0}
💰 Jami xarajat: ${stats_data['total_cost'] or 0:.4f}"""

    await update.message.reply_text(message)


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bugungi statistika"""
    if not await check_is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat adminlar uchun!")
        return

    total, answered, not_found, stats_data = await get_today_stats()

    message = f"""📊 Bugungi statistika:

💬 Savollar: {total}
✅ Javob berilgan: {answered}
❌ Javob topilmagan: {not_found}

🔢 Tokenlar: {stats_data['total_tokens'] or 0}
💰 Xarajat: ${stats_data['total_cost'] or 0:.4f}"""

    await update.message.reply_text(message)


async def unanswered(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Javob berilmagan savollar"""
    if not await check_is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat adminlar uchun!")
        return

    not_found_convs = await get_unanswered_convs()

    if not not_found_convs:
        await update.message.reply_text("Javob berilmagan savollar yo'q!")
        return

    message = "❌ Javob berilmagan savollar:\n\n"
    for conv in not_found_convs:
        message += f"👤 {conv.user.username or conv.user.telegram_id}\n"
        message += f"❓ {conv.question[:100]}\n"
        message += f"📅 {conv.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"

    await update.message.reply_text(message)


async def costs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xarajatlar hisoboti"""
    if not await check_is_admin(update.effective_user.id):
        await update.message.reply_text("Bu buyruq faqat adminlar uchun!")
        return

    today_cost, week_cost, month_cost, total_cost = await get_costs_stats()

    message = f"""💰 Xarajatlar hisoboti:

📅 Bugun: ${today_cost:.4f}
📅 Bu hafta: ${week_cost:.4f}
📅 Bu oy: ${month_cost:.4f}
📅 Jami: ${total_cost:.4f}"""

    await update.message.reply_text(message)


def main():
    """Bot ishga tushirish"""
    app = Application.builder().token(settings.BOT_TOKEN).build()

    # User handlerlar
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))

    # Admin handlerlar
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("unanswered", unanswered))
    app.add_handler(CommandHandler("costs", costs))

    print("Bot ishlamoqda...")
    app.run_polling()


if __name__ == "__main__":
    main()
