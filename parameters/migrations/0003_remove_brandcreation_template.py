# Generated by Django 5.1.2 on 2024-10-19 16:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('parameters', '0002_remove_brandcreation_cta_required_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='brandcreation',
            name='template',
        ),
    ]
