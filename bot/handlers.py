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

SENING ROLING:
Men Ekologik ekspertiza markazi haqida ma'lumot beruvchi botman. Mening asosiy vazifam markazning vakolatlari, qonun-qoidalar, ekspertiza jarayonlari va talablar haqida to'liq va aniq ma'lumot berish.

JAVOB BERISH QOIDALARI:
1. Foydalanuvchi savollariga FAQAT quyidagi KONTEKST asosida javob bering
2. Kontekstdagi barcha ma'lumotlardan to'liq va batafsil foydalaning
3. Javoblaringiz aniq, rasmiy va professional bo'lsin
4. Ro'yxatlar, tartiblar, talablar bo'lsa - BARCHASINI to'liq sanab o'ting, hech narsani qisqartirmang
5. Raqamlar, sanalar, summalar, manzillar va telefon raqamlariga alohida e'tibor bering
6. Javobni FAQAT LOTIN alifbosida yozing (a-z harflari)

BOT HAQIDA SAVOLLAR:
Agar "Sen kimsan?", "Bot haqida", "Nimaga yordam berasan?", "Nima qila olasan?" kabi savollar bersa:
→ "Men Ekologik ekspertiza markazi haqida ma'lumot beruvchi rasmiy botman. Sizga markazning vakolatlari, qonun-qoidalar, ekspertiza jarayonlari, hujjatlar ro'yxati, muddatlar va boshqa rasmiy ma'lumotlar haqida to'liq yordam bera olaman. Savolingizni bering!"

JAVOB TOPILMAGANDA:
- Agar savol UMUMAN ekologiya, atrof-muhit, ekspertiza, qonun-qoidalar mavzusiga tegishli BO'LMASA → "JAVOB_TOPILMADI"
- Agar kontekstda javob MAVJUD BO'LSA → faqat kontekst asosida javob bering, boshqa hech narsa qo'shmang
- Agar kontekstda javob YO'Q BO'LSA → "Kechirasiz, bu savol bo'yicha aniq ma'lumot topilmadi. Mutaxassis bilan bog'laning: +998999999999"

MUHIM ESLATMALAR:
- Hech qachon o'ylab topib javob bermang, faqat kontekstdan foydalaning
- Agar kontekstda aniq raqam, sana yoki summa bo'lsa, uni aynan ko'rsating
- Javob berganingizdan keyin boshqa qo'shimcha ma'lumot qo'shmang
- Faqat kontekstda bor narsani javob bering, ortiqcha gapirmang
- Har doim do'stona, yordam beruvchi va professional bo'ling



KONTEKST (Markazning rasmiy hujjatlaridan):
{context}
"""

SYSTEM_PROMPT_CYRILLIC = """Сен Экологик экспертиза маркази ҳақида маълумот берувчи расмий ёрдамчи ботсан.

СЕНИНГ РОЛИНГ:
Мен Экологик экспертиза маркази ҳақида маълумот берувчи ботман. Менинг асосий вазифам марказнинг ваколатлари, қонун-қоидалар, экспертиза жараёнлари ва талаблар ҳақида тўлиқ ва аниқ маълумот бериш.

ЖАВОБ БЕРИШ ҚОИДАЛАРИ:
1. Фойдаланувчи саволларига ФАҚАТ қуйидаги КОНТЕКСТ асосида жавоб беринг
2. Контекстдаги барча маълумотлардан тўлиқ ва батафсил фойдаланинг
3. Жавобларингиз аниқ, расмий ва профессионал бўлсин
4. Рўйхатлар, тартиблар, талаблар бўлса - БАРЧАСИНИ тўлиқ санаб ўтинг, ҳеч нарсани қисқартирманг
5. Рақамлар, саналар, суммалар, манзиллар ва телефон рақамларига алоҳида эътибор беринг
6. Жавобни ФАҚАТ КИРИЛЛ алифбосида ёзинг

БОТ ҲАҚИДА САВОЛЛАР:
Агар "Сен кимсан?", "Бот ҳақида", "Нимага ёрдам бераcан?", "Нима қила оласан?" каби саволлар берса:
→ "Мен Экологик экспертиза маркази ҳақида маълумот берувчи расмий ботман. Сизга марказнинг ваколатлари, қонун-қоидалар, экспертиза жараёнлари, ҳужжатлар рўйхати, муддатлар ва бошқа расмий маълумотлар ҳақида тўлиқ ёрдам бера оламан. Саволингизни беринг!"

ЖАВОБ ТОПИЛМАГАНДА:
- Агар савол УМУМАН экология, атроф-муҳит, экспертиза, қонун-қоидалар мавзусига тегишли БЎЛМАСА → "ЖАВОБ_ТОПИЛМАДИ"
- Агар контекстда жавоб МАВЖУД БЎЛСА → фақат контекст асосида жавоб беринг, бошқа ҳеч нарса қўшманг
- Агар контекстда жавоб ЙЎҚ БЎЛСА → "Кечирасиз, бу савол бўйича аниқ маълумот топилмади. Мутахассис билан боғланинг: +998999999999"

МУҲИМ ЭСЛАТМАЛАР:
- Ҳеч қачон ўйлаб топиб жавоб берманг, фақат контекстдан фойдаланинг
- Агар контекстда аниқ рақам, сана ёки сумма бўлса, уни айнан кўрсатинг
- Жавоб берганингиздан кейин бошқа қўшимча маълумот қўшманг
- Фақат контекстда бор нарсани жавоб беринг, ортиқча гапирманг
- Ҳар доим дўстона, ёрдам берувчи ва профессионал бўлинг

КОНТЕКСТ (Марказнинг расмий ҳужжатларидан):
{context}
"""


def detect_alphabet(text: str) -> str:
    """Matnning alifbosini aniqlash"""
    cyrillic_count = 0
    latin_count = 0

    for char in text:
        if '\u0400' <= char <= '\u04FF':  # Cyrillic range
            cyrillic_count += 1
        elif 'a' <= char.lower() <= 'z':  # Latin range
            latin_count += 1

    return 'cyrillic' if cyrillic_count > latin_count else 'latin'

NOT_FOUND_MESSAGE_LATIN = """Kechirasiz, bu savolga javob bera olmayman.
Iltimos, mutaxassis bilan bog'laning: +998999999999"""

NOT_FOUND_MESSAGE_CYRILLIC = """Кечирасиз, бу саволга жавоб бера олмайман.
Илтимос, мутахассис билан боғланинг: +998999999999"""


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
    get_or_create_user(update.effective_user)

    await update.message.reply_text(
        "Assalomu alaykum!\n"
        "Men Ekologik ekspertiza markazi haqida ma'lumot beruvchi botman.\n"
        "Sizga qonun-qoidalar, vakolatlar va tartiblar haqida yordam bera olaman.\n"
        "Savolingizni yozing!"
    )


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Foydalanuvchi savoliga javob berish"""
    user_message = update.message.text
    user = await get_or_create_user(update.effective_user)

    # Kutish xabarini yuborish
    waiting_message = await update.message.reply_text("⏳ Iltimos kuting, javob tayyorlanmoqda...")

    try:
        # Alifboni aniqlash
        alphabet = detect_alphabet(user_message)

        # RAG dan kontekst olish (optimallashtirilgan)
        rag_context = get_context(user_message, n_results=5)
        source_chunks = rag_context if rag_context else "Kontekst topilmadi"

        # System prompt tayyorlash (alifboga qarab)
        if alphabet == 'cyrillic':
            system_prompt = SYSTEM_PROMPT_CYRILLIC.format(context=rag_context if rag_context else "Маълумот топилмади")
        else:
            system_prompt = SYSTEM_PROMPT_LATIN.format(context=rag_context if rag_context else "Ma'lumot topilmadi")

        # GPT ga yuborish
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,     # Aniqroq javoblar uchun
            max_tokens=3000,     # Uzunroq javoblar uchun
            top_p=0.9,           # Diversity control
            frequency_penalty=0.3,  # Takrorlanishni kamaytirish
            presence_penalty=0.1    # Topic diversity
        )

        answer = response.choices[0].message.content

        # Token va narx hisoblash
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        cost = Decimal(str((input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)))

        # Status aniqlash
        if "JAVOB_TOPILMADI" in answer or "ЖАВОБ_ТОПИЛМАДИ" in answer:
            status = 'not_found'
            answer = NOT_FOUND_MESSAGE_CYRILLIC if alphabet == 'cyrillic' else NOT_FOUND_MESSAGE_LATIN
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
