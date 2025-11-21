from django.db import models


class TelegramUser(models.Model):
    telegram_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, blank=True, null=True)
    first_name = models.CharField(max_length=255, blank=True, null=True)
    last_name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Telegram User"
        verbose_name_plural = "Telegram Users"

    def __str__(self):
        return f"{self.username or self.telegram_id}"


class Conversation(models.Model):
    STATUS_CHOICES = [
        ('answered', 'Javob berildi'),
        ('not_found', 'Javob topilmadi'),
    ]

    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE, related_name='conversations')
    question = models.TextField()
    answer = models.TextField()
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='answered')
    source_chunks = models.TextField(blank=True, null=True)  # RAG dan topilgan qismlar
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Conversation"
        verbose_name_plural = "Conversations"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user} - {self.question[:50]}"


class BotAdmin(models.Model):
    telegram_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Bot Admin"
        verbose_name_plural = "Bot Admins"

    def __str__(self):
        return f"{self.username or self.telegram_id}"
