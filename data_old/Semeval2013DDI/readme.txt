This zip contains the evaluation scripts for the Semeval-2013 Task 9. There are two scripts in this package:
- evaluateNER.jar : evaluation script for the task 9.1 (recognition and classification of pharmacological substances)

- evaluateDDI.jar:  evaluation script for the task 9.2 (extraction of drug-drug interactions)

For a detailed description of the task, please see the task homepage

http://www.cs.york.ac.uk/semeval-2013/task9/

http://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/task-9.1-drug-ner.pdf

http://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/task-9.2-ddi-extraction.pdf

Please send an email to isegura@inf.uc3m.es to report any error.

----------

Usage:


To perform evaluation of the task 9.1, the gold standerd dataset used for the system evaluation must be saved into a directory (goldDir). Then, you should run the following command:

    java -jar evaluateNER.jar <goldDir> <submissionFile>

where 
- goldDir is the directory where the gold standard dataset has been saved, and
- submissionFile is a file containing your predicted annotations for this dataset (for example, task9.1_UC3M_1.txt)


To perform evaluation of the task 9.2, the gold standard dataset used for the system evaluation must be saved into a directory (goldDir).  This dataset must follow the format with pair tags (http://www.cs.york.ac.uk/semeval-2013/task9/index.php?id=data). Then, you should run the following command:

    java -jar evaluateDDI.jar <goldDir> <submissionFile>

where 
- goldDir is the directory where the gold standard dataset has been saved, and
- submissionFile is a txt file containing your predicted annotations for this dataset (for example, task9.2_UC3M_1.txt)


----------

This software is copyright (c) DDIExtraction 2013 shared task organizers.




