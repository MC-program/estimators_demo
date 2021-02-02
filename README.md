# estimators_demo

This is a demo of several estimators. The main module runs a list of them, the winner
will be selected and fed the entire dataset to fit this model.
Then the validation set can be used to predict unseen output data.
This data is saved under validation_prediction.csv.
A plot to each model is exported to plot.pdf.

Start with:
Python main.py training.xml validation.xml

training.xml is the training set.
validation.xml is for test only and cannot be used to parameterize.



Example (one iteration of main) of validation_prediction.csv:
entry_n;energy
entry_2;-0.26204
entry_16;-0.15486
entry_28;-0.12822
entry_40;-0.23362
entry_44;-0.2706
entry_49;-0.18583999999999998
entry_61;-0.5247200000000001
entry_68;-0.08621999999999999
entry_74;-0.20115999999999995
entry_75;-0.1772
entry_90;-0.20477999999999996
entry_126;-0.08412
entry_132;-0.20146000000000003
entry_141;-0.26064000000000004
entry_157;-0.16472
entry_168;-0.16776
entry_169;-0.03784
entry_176;-0.25422
entry_191;-0.19388
entry_198;-0.28578000000000003
entry_202;-0.17195999999999997
entry_205;-0.13163999999999998
