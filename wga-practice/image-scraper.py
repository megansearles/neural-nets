import csv
import urllib

with open('wga-sample.txt', 'rb') as f:
	r = csv.reader(f, delimiter=';')
	for row in r:
		url1 = row[6]
		# use urlparse to get the components of the url
		# use urlretrieve to download image