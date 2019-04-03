# mortgage_default_model
Project to compare various models for mortgage default prediction.

For the data set, we want to start with all of the acquisition data, the 
number of 0, 1, 2, 3 month delinquencies, whether the loan was foreclosed. 
To determine whether the property was foreclosed on, we are going to use the
foreclosure date. It's worth noting that this is the date the property went to
sale. This DOES NOT work for REO property. Likely in the future, we will need
to update the logic for this to include REO. 

## TODO:
* Update Extractor module to take arguments. For example, we need an arg for
whether to remove the zip files when we extract. 
* Automate data pull from FNMA. Can potentially be done with scrapy. 
* Pythonize repo - add "required", etc.
* Validate data transformations.
* Specify data types in config file. Will have to parse columns vs. types
in code.
* Refactor extract_performance_counts - method too long. 
* Try with a few different model types. Document the things I've tried already,
and replicate them.

## Resources
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_File_layout.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/performance-sample-file.txt
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_Glossary.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_FAQs.pdf
* http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html

## References
Data Obtained From FNMA and not published through this project.
