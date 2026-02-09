"""
Management command to create static website user
"""
from django.core.management.base import BaseCommand
from bot.models import TelegramUser


class Command(BaseCommand):
    help = 'Website uchun static user yaratish'

    def handle(self, *args, **options):
        """Website user yaratish yoki mavjudligini tekshirish"""
        user, created = TelegramUser.objects.get_or_create(
            telegram_id=-1,  # Website uchun maxsus ID (-1 hech qachon haqiqiy telegram ID emas)
            defaults={
                'username': 'website_user',
                'first_name': 'Website',
                'last_name': 'User'
            }
        )

        if created:
            self.stdout.write(self.style.SUCCESS('✅ Website user muvaffaqiyatli yaratildi!'))
            self.stdout.write(f'   - telegram_id: {user.telegram_id}')
            self.stdout.write(f'   - username: {user.username}')
            self.stdout.write(f'   - ID: {user.id}')
        else:
            self.stdout.write(self.style.WARNING('⚠️  Website user allaqachon mavjud!'))
            self.stdout.write(f'   - telegram_id: {user.telegram_id}')
            self.stdout.write(f'   - username: {user.username}')
            self.stdout.write(f'   - ID: {user.id}')
