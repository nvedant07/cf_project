# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-11-13 18:26
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('src', '0002_auto_20171113_1738'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie',
            name='image',
            field=models.FilePathField(blank=True, max_length=1000, path='/Users/vedant/Desktop/CF/cf_project/imdb_rater/movie_posters'),
        ),
        migrations.AlterField(
            model_name='movie',
            name='name',
            field=models.CharField(max_length=100),
        ),
    ]
