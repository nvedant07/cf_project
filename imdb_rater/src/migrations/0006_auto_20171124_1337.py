# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-24 13:37
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('src', '0005_dislikedfilms'),
    ]

    operations = [
        migrations.RenameField(
            model_name='feedback',
            old_name='count',
            new_name='dislike_count',
        ),
    ]
