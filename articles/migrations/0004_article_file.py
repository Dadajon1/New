# Generated by Django 3.1 on 2021-08-17 06:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('articles', '0003_article_photo2'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='file',
            field=models.FileField(blank=True, upload_to='files/'),
        ),
    ]