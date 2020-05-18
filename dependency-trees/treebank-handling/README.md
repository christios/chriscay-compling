# Prague Dependency Treebank (PDT)

## Error Checker
In this `pdt_error_checker.py`, the following is done:
1. Checking for syntax errors
2. Checking for semantic errors
3. Converting the trees from a referential to a structural notation (using parentheses)

Note:

* If there are some syntax errors, the program may throw an error while checking for semantic errors in step 2, so any syntax error should be fixed by the user for the program to accurately diagnose all semantic errors.

* In some rare cases, for the program to accurately diagnose all semantic errors, the already diagnosed semantic errors should be fixed first. So in some sense this is a debugger-like program and requires the interaction of the user to find all possible errors.

* The command `make get_data` can be used to download and unzip the data.

* The command `python3 hw1.py file_name [p]`
can be used to run the program. If `p` (or any other word/letter) is written after `file_name`, then the result of the conversion will be outputted to the command line. In both cases, the conversion can be found in a file called `file_name.conv` in the same directory as `file_name`.

## Form change extraction
In this implementation, the following is done:

1. The zipped folder is downloaded if it is not already present in the current directory
2. Form changes are extracted from the m-layer files
3. Corresponding tokens are then extracted from the w-layer files (if there are any)
4. If the there is a form change between the word and mophological layers (as can sometimes happen when tokens are split or joined, etc.), then the change is printed 

Note: no unzipping is required beforehand as the code works with the main folder as it is.

To run the file type the command `python3 form_change.py`.

# Universal Dependencies

The file `assignprontype.py` is a Udapy module which assigns missing `PronType` (Universal Dependencies 2.5) features to words in a French treebank.