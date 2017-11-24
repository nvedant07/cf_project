from django import forms

RATINGS = [('1','1'),('2','2'),('3','3'),('4','4'),('5','5')]
TRUSTS = [('1','Yes'),('2','No')]
ALGORITHMS = [('1', 'Algo A'),('2', 'Algo B'),('3', 'Algo C')]

class RatingForm(forms.Form):
	rating = forms.ChoiceField(choices=RATINGS, widget=forms.RadioSelect())

class TrustForm(forms.Form):
	pass
	# trust = forms.ChoiceField(choices=TRUSTS, widget=forms.RadioSelect())

class AlgoForm(forms.Form):
	algo = forms.ChoiceField(choices=ALGORITHMS, widget=forms.RadioSelect())
