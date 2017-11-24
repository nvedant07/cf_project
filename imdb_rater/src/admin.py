# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from models import *
# Register your models here.

class MovieAdmin(admin.ModelAdmin):
	list_display = ['movielens_id', 'name', 'url', 'image']
admin.site.register(Movie, MovieAdmin)

class UserRatingAdmin(admin.ModelAdmin):
	list_display = ['corresponding_user', 'movie', 'rating']
admin.site.register(UserRating, UserRatingAdmin)

class TrustAdmin(admin.ModelAdmin):
	list_display = ['issuing_user', 'trusted_user', 'trust_value']
admin.site.register(Trust, TrustAdmin)

class FeedbackAdmin(admin.ModelAdmin):
	list_display = ['algorithm', 'algorithm_name', 'dislike_count']
	def algorithm_name(self, obj):
		if int(obj.algorithm) == 1:
			return 'User-User'
		elif int(obj.algorithm) == 2:
			return 'Item-Item'
		else:
			return 'Our Model'
admin.site.register(Feedback, FeedbackAdmin)

class DislikedFilmsAdmin(admin.ModelAdmin):
	list_display = ['user', 'movie']
admin.site.register(DislikedFilms, DislikedFilmsAdmin)