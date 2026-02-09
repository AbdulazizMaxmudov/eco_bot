from django.core.management.base import BaseCommand
from django.conf import settings
import os
from rag.vectordb import index_rules


class Command(BaseCommand):
    help = 'Rules faylni vector DB ga indekslash'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='rules.docx',
            help='Rules fayl nomi (default: rules.docx)'
        )
        parser.add_argument(
            '--lang',
            type=str,
            default='uz',
            choices=['uz', 'ru'],
            help='Til: uz yoki ru (default: uz)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Barcha tillarni indekslash (rules.docx + rules_ru.docx)'
        )

    def handle(self, *args, **options):
        if options['all']:
            # Barcha tillarni indekslash
            files = [
                ('rules.docx', 'uz'),
                ('rules_ru.docx', 'ru'),
            ]
            for file_name, lang in files:
                file_path = os.path.join(settings.BASE_DIR, file_name)
                if not os.path.exists(file_path):
                    self.stdout.write(self.style.WARNING(f"Fayl topilmadi: {file_path} - o'tkazildi"))
                    continue
                self.stdout.write(f"Indekslash: {file_path} ({lang})")
                try:
                    index_rules(file_path, lang=lang)
                    self.stdout.write(self.style.SUCCESS(f'{file_name} ({lang}) muvaffaqiyatli indekslandi!'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Xatolik ({file_name}): {e}"))
        else:
            file_name = options['file']
            lang = options['lang']
            file_path = os.path.join(settings.BASE_DIR, file_name)

            if not os.path.exists(file_path):
                self.stdout.write(self.style.ERROR(f"Fayl topilmadi: {file_path}"))
                return

            self.stdout.write(f"Indekslash boshlanmoqda: {file_path} ({lang})")

            try:
                index_rules(file_path, lang=lang)
                self.stdout.write(self.style.SUCCESS('Indekslash muvaffaqiyatli yakunlandi!'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Xatolik: {e}"))
