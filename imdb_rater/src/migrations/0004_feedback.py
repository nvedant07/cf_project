# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-16 02:55
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('src', '0003_auto_20171113_1826'),
    ]

    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('algorithm', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('count', models.CharField(blank=True, max_length=10000)),
            ],
        ),
    ]
