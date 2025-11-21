from django.contrib import admin
from django.db.models import Sum, Count
from django.utils.html import format_html
from .models import TelegramUser, Conversation, BotAdmin
import csv
from django.http import HttpResponse


@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = ['telegram_id', 'username', 'first_name', 'last_name', 'conversation_count', 'created_at']
    search_fields = ['telegram_id', 'username', 'first_name', 'last_name']
    list_filter = ['created_at']
    readonly_fields = ['created_at']

    def conversation_count(self, obj):
        return obj.conversations.count()
    conversation_count.short_description = 'Savollar soni'


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'short_question', 'status_badge', 'total_tokens', 'cost_display', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['question', 'answer', 'user__username', 'user__telegram_id']
    readonly_fields = ['created_at', 'input_tokens', 'output_tokens', 'total_tokens', 'cost']
    date_hierarchy = 'created_at'
    actions = ['export_to_csv']

    def short_question(self, obj):
        return obj.question[:50] + '...' if len(obj.question) > 50 else obj.question
    short_question.short_description = 'Savol'

    def status_badge(self, obj):
        if obj.status == 'answered':
            return format_html('<span style="color: green;">✓ Javob berildi</span>')
        return format_html('<span style="color: red;">✗ Topilmadi</span>')
    status_badge.short_description = 'Status'

    def cost_display(self, obj):
        return f"${obj.cost:.6f}"
    cost_display.short_description = 'Narx'

    def export_to_csv(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="conversations.csv"'
        response.write('\ufeff'.encode('utf8'))  # BOM for Excel

        writer = csv.writer(response)
        writer.writerow(['ID', 'User', 'Savol', 'Javob', 'Status', 'Tokens', 'Narx', 'Sana'])

        for conv in queryset:
            writer.writerow([
                conv.id,
                conv.user.username or conv.user.telegram_id,
                conv.question,
                conv.answer,
                conv.status,
                conv.total_tokens,
                f"${conv.cost:.6f}",
                conv.created_at.strftime('%Y-%m-%d %H:%M')
            ])

        return response
    export_to_csv.short_description = "CSV ga eksport qilish"


@admin.register(BotAdmin)
class BotAdminAdmin(admin.ModelAdmin):
    list_display = ['telegram_id', 'username', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['telegram_id', 'username']


# Dashboard statistika
class DashboardAdmin(admin.AdminSite):
    site_header = "Eco Bot Admin"
    site_title = "Eco Bot"
    index_title = "Boshqaruv paneli"
