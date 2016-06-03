import os
import csv
import urllib

#print os.listdir("../..")

with open('wga-sample.txt', 'rb') as f:
	r = csv.reader(f, delimiter=';')
	r.next()
	os.chdir("Art/a/aachen")
	for row in r:
		
		# Change url in file to url with just the picture
		url1 = row[6]
		url2 = url1.replace("html","art",1)
		url2 = url2.replace("html","jpg")
		
		# Stores list of components of url, along with specifics needed for saving images
		url_comp = url2.split('/')
		title = url_comp[-1]
		artist = url_comp[-2]
		letter = url_comp[-3]
		
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
		# Something wrong in above code that made it so that the first "b"
		# artist's folder was saved in the letter directory instead of "b"