import csv

with open('wga-sample.txt', 'rb') as wga_file:
	wga_reader = csv.reader(wga_file, delimiter=';')
	for row in wga_reader:
		print row