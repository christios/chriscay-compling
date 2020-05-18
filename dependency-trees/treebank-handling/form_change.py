#!/usr/bin/env python3
import os
import urllib.request
import zipfile
import gzip
import re
from io import BytesIO

URL = "http://ufal.mff.cuni.cz/~mirovsky/vyuka/NPFL075/2020/02/train-1.zip"

def detect_change (m_data=""):
	""" This function detects if at least one form change occured, and returns
		all these form changes as a dictionary with keys corresponding to <w.rf>,
		and values as 2-tuples containing the contents of <form_change> and and <form>.
	"""
	form_changes = {}
	# Remove leading tabs and store each line as an entry in a list
	m_data = [line.lstrip() for line in m_data.splitlines()]
	# Detect all form_change elements
	for line_no, line in enumerate(m_data):
		match = re.match(r"<form_change>(.+)</form_change>", line)
		# If a form_change is found, store all the necessary information in the form_changes dictionary
		if match:
			form_change = match.group(1)
			form = m_data[line_no+1][6:-7]
			w_ref = m_data[line_no-2][8:-5]
			form_changes[w_ref] = (form_change, form)
	return form_changes

if __name__ == '__main__':
	# Extract the name of the zip file
	path = os.path.basename(URL)
	# Download file if it is not already in the dir
	if not os.path.exists(path):
		print("Downloading data {}...".format(path))
		urllib.request.urlretrieve(URL, filename=path)

	# Read into the main zip file
	with zipfile.ZipFile(path, 'r') as zip_file:
		# Access each folder in the main file
	    for name in zip_file.namelist():
	    	# If the file is an m-layer file...
	    	if name[-4] == 'm':
	    		# ... save it, read into it, and extract form changes
	    		zip_file_m_data = BytesIO(zip_file.read(name))
	    		with gzip.open(zip_file_m_data, 'r') as fh_m:
	    			m_data = fh_m.read().decode("utf-8")
	    			form_changes = detect_change(m_data)
	    			# If there are any form changes
	    			if form_changes:
	    				# Retrieve the corresponding w-layer file path
	    				w_ref_path = re.search(r"<reffile.+href=\"(.+)\".+>", m_data).group(1)
	    				# Save the corresponding w-layer file, open it, and read into it
	    				zip_file_w_data = BytesIO(zip_file.read("{}/{}".format(os.path.splitext(path)[0], w_ref_path)))
	    				with gzip.open(zip_file_w_data, 'r') as fh_w:
	    					w_data = fh_w.read().decode("utf-8")
	    					# Gather the corresponding word forms and print all the information
	    					for w_ref, info in form_changes.items():
	    						# If the form_change is insert, then only print the m-layer form
	    						if info[0] != "insert":
		    						token = re.search(w_ref + r"\">\s+<token>(.+)<", w_data).group(1)
		    						print("{:<23}{:<20}{:<25}{}".format(w_ref, info[0], token, info[1]))
		    					else:
		    						print("{:<23}{:<20}{:<25}{}".format(w_ref, info[0], "", info[1]))