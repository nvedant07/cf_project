# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.conf import settings
import os

from django.contrib.auth.models import User
from django.utils.encoding import python_2_unicode_compatible
# Create your models here.

@python_2_unicode_compatible
class Movie(models.Model):
	movielens_id = models.CharField(max_length=100, primary_key=True)
	name = models.CharField(max_length=100)
	url = models.CharField(max_length=100)
	image = models.FilePathField(max_length=1000, blank=True, path=os.path.join(settings.BASE_DIR, "movie_posters"))
	def __str__(self):
		return self.name

class UserRating(models.Model):
	corresponding_user = models.ForeignKey(User)
	movie = models.ForeignKey(Movie)
	rating = models.CharField(max_length=100, default='0')
	def __str__(self):
		return self.corresponding_user.first_name + " " + self.corresponding_user.last_name

class Trust(models.Model):
	issuing_user = models.ForeignKey(User, null=True, related_name='issuing_user')
	trusted_user = models.ForeignKey(User, null=True, related_name='trusted_user')
	trust_value = models.CharField(max_length=10, blank=True)

class Feedback(models.Model):
	algorithm = models.CharField(max_length=10, primary_key=True)
	dislike_count = models.CharField(max_length=10000, blank=True)

class DislikedFilms(models.Model):
	user = models.ForeignKey(User)
	movie = models.ForeignKey(Movie)