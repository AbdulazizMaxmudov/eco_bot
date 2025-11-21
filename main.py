from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from docx import Document
from dotenv import load_dotenv
import os

# .env fayldan o'qish
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Word fayldan qoidalarni o'qish
def load_rules():
    doc = Document("rules.docx")
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Qoidalarni bir marta yuklash
RULES_TEXT = load_rules()

# System prompt
SYSTEM_PROMPT = f"""Sen Ekologik ekspertiza markazi haqida ma'lumot beruvchi yordamchi botsan.

Sening vazifang:
1. Foydalanuvchi savollariga FAQAT quyidagi qonun-qoidalar asosida javob berish
2. Agar savol qoidalarda bo'lmasa, "Kechirasiz, bu savolga javob bera olmayman. Iltimos, mutaxassis bilan bog'laning: +998999999999" deb javob ber
3. Foydalanuvchi qaysi alifboda (lotin yoki kirill) yozsa, shu alifboda javob ber
4. Javoblar aniq, qisqa va tushunarli bo'lsin
5. Agar foydalanuvchi bot haqida so'rasa: "Bu bot Ekologik ekspertiza markazining vakolatlari va qonun-qoidalari haqida savollarga javob beradi"

Qonun-qoidalar:
{RULES_TEXT}
"""

# /start buyrug'i
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Assalomu alaykum! 👋\n"
        "Men Ekologik ekspertiza markazi haqida ma'lumot beruvchi botman.\n"
        "Sizga qonun-qoidalar, vakolatlar va tartiblar haqida yordam bera olaman.\n"
        "Savolingizni yozing!"
    )

# AI orqali javob berish
async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    try:
        # GPT ga yuborish
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Token hisob-kitobi
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Narx hisoblash (GPT-4o-mini)
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        print(f"Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}, Cost: ${cost:.6f}")

        await update.message.reply_text(answer)

    except Exception as e:
        print(f"Xatolik: {e}")
        await update.message.reply_text(
            "Kechirasiz, texnik xatolik yuz berdi. Iltimos, keyinroq urinib ko'ring yoki mutaxassis bilan bog'laning:\n"
            "+998999999999"
        )

# Bot ishga tushirish
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Handlerlar
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))

    print("Bot ishlamoqda...")
    app.run_polling()

if __name__ == "__main__":
    main()
