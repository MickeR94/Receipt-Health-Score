### Health Score for ICA Receipts

This is an AI project from the course DI1214 at KTH.
The project is duplicated from my private KTH repo. All personal ICA receipts that are required to run this program are removed for privacy reasons.
If the reader would like to test the functionality, it is recommended to upload their own ICA receipts in PDF-format in the specified directory.

### About this project

I came up with this project when I wanted to get a better grasp of my spending habits on ICA, and to see if my purchases are in line with my fitness goals (HINT: they're not).
Hence, I made this as a final school assignment for the Artificial intelligence course ID1214.

It's a simple program that parses the text using `pdfplumber` and extract the relevant information. In this case, that information is the grocery item and corresponing price.
The original intention was to calculate total money spent, as well as total money spent on junk food/energy drinks. This idea is put on hold as of now (2026-06-14) as it was more related to dataprocessing rather than AI, which is the scope of the course.

The program uses Zero-Shot-classification, more specifically `megatron-bert-large-swedish-cased-165-zero-shot`, see https://kb-labb.github.io/posts/2023-02-12-zero-shot-text-classification/.
It is a Natural Language Inference-model that required no prior training. This was suitable for this small project as it saved time by removing the need of having to manually label a training set.

Each grocery item recieves a label which corresponds to a health score: Hälsosam mat: 1.0, Skräpmat: 0.0, Energidryck: 0.0, Hushåll: 0.5. Once wach item is labeled, the average health score is calculated.

The end result works okay, but it certainly could be better. Regular swedish words such as "Bearnaisesås" is classified correclty, but as soon as the item's name deviates slightly from the proper spelling, the classification accuracy decreases. 
Same with items named after flavors, such as "Havssalt och Gräslök". This is a flavor of chips, but is classified as "Hälsosam mat", possibly due to "Gräslök". 

### Future Work

There's much more that would be cool to develop for this.
First of all, improving the classification would be great. Zero-Shot-classification uses `sequences` and seem to do better with full sentences. Now with only one or a few words, the classification accuracy seems to suffer. 
One solution could be to test few-shot-classification instead and take the time to write a .csv-file with ground truth labels.

Another thing that would be great is a GUI, which would of course look nice but it could also display more info, such as monthly spending, how much money has gone to energy drinks, junk food, and healthy food, if the user is within budget etc.

I'd also like to expand this beyond only ICA recipts in PDF-format. It would be great to be able to take a photo of a physical receipt and use OCR for the text extraction. It would make the program more versitile and useful if it is not limited to only ICA.
