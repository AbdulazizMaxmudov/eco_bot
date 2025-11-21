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

    def handle(self, *args, **options):
        file_name = options['file']
        file_path = os.path.join(settings.BASE_DIR, file_name)

        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f"Fayl topilmadi: {file_path}"))
            return

        self.stdout.write(f"Indekslash boshlanmoqda: {file_path}")

        try:
            index_rules(file_path)
            self.stdout.write(self.style.SUCCESS('Indekslash muvaffaqiyatli yakunlandi!'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Xatolik: {e}"))
