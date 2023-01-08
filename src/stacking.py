from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier,AdaBoostClassifier,BaggingClassifier

estimators = [('et', ExtraTreesClassifier(n_estimators=20,max_depth=None, min_samples_split=2, random_state=0)), ('svc', SVC(random_state=0))]

my_clf = StackingClassifier(estimators=estimators,final_estimator=BaggingClassifier(AdaBoostClassifier(random_state=0),n_estimators=10,random_state=0))
