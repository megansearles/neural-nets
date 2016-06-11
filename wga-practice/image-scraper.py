''' To-Do Before Running the Program the First Time:
		1. Create file "pickup-place.txt" which only contains the number 0
		2. Create folders "Art/a/aachen"
		3. Change "wga-sample.txt" to "catalog.txt"
'''

import os
import csv
import urllib

new_catalog = []
pickup_place = len(new_catalog)

# If program had stopped previously, this will allow it to start where it left off
with open('pickup-place.txt', 'rb') as f:
	pickup_place = f.read()

old_place = pickup_place

with open('wga-sample.txt', 'rb') as f:
	r = csv.reader(f, delimiter=';')
	r.next()
	os.chdir("Art/a/aachen")
	
	# Skip to next line
	if pickup_place != 0:
		for n in xrange(int(pickup_place)):
			r.next()
	
	for row in r :
		
		# Change url in file to url with just the picture
		url1 = row[6]
		url2 = url1.replace("html","art",1)
		url2 = url2.replace("html","jpg")
		
		# Stores list of components of url, along with specifics needed for saving images
		url_comp = url2.split('/')
		title = url_comp[-1]
		letter = url_comp[4]
		artist = url_comp[5]
		
		# Saves lists of directories for artists and letters
		a_dir = os.listdir("..")
		l_dir = os.listdir("../..")
		
		# Change directory to letter and artist folder and save image
		if artist == a_dir[-1]:
			os.chdir("../" + artist)
			urllib.urlretrieve(url2,title)
		elif letter == l_dir[-1]:
			os.chdir("../../" + letter)
			a_dir = os.listdir(".")
			if artist not in a_dir:
				os.mkdir(artist)
			os.chdir(artist)
			urllib.urlretrieve(url2,title)
		else:
			os.chdir("../..")
			os.mkdir(letter)
			os.chdir(letter)
			os.mkdir(artist)
			os.chdir(artist)
			urllib.urlretrieve(url2,title)
		
		# Update row to include file path instead of url and add onto a running list of all the rows
		new_path = "Art" + "/" + letter + "/" + artist + "/" + title
		row[6] = new_path
		new_catalog.append(row)
		
		# Update the new catalog file
		with open('../../../updated-sample.txt', 'ab') as f:
			w = csv.writer(f, delimiter=';')
			w.writerow(row)
			
		# Update file with the next row number to pick up on
		pickup_place = len(new_catalog) + int(old_place)
		with open('../../../pickup-place.txt', 'wb') as f:
			f.write(str(pickup_place))