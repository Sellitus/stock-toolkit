import bulbea as bb

share = bb.Share('YAHOO', 'GOOGL')

Xtrain, Xtest, ytrain, ytest = bb.learn.evaluation.split(share, 'Close', normalize=True)

import pdb
pdb.set_trace()