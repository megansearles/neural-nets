import os
import csv
import urllib

new_catalog = []

with open('wga-sample.txt', 'rb') as f:
	r = csv.reader(f, delimiter=';')
	r.next()
	os.chdir("Art/a/aachen") # Don't forget to make these folders before running this on the actual catalog
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
			urllib.urlretrieve(url2,title)
		elif letter == l_dir[-1]:
			os.chdir("..")
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
		new_path = "Art" + "/" + letter + "/" + artist + "/" + title # (this may not be the desired format - check)
		row[6] = new_path
		new_catalog.append(row)
		
		# Update the new catalog file
		with open('../../../updated-sample.txt', 'ab') as f:
			w = csv.writer(f, delimiter=';')
			w.writerow(row)
			
		# Update file with the next row number to pick up on
		pickup_place = len(new_catalog)
		with open('../../../pickup-place.txt', 'wb') as f:
			f.write(str(pickup_place))