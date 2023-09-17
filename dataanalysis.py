import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score
import feature_engine
from feature_engine.selection import DropCorrelatedFeatures
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import pearsonr
import imblearn
from imblearn.over_sampling import SMOTE
import logging
logging.basicConfig(filemode='w',filename='Creditcard.log',format='%(asctime)s [%(levelname)s] - %(message)s',level=logging.DEBUG)

class Model_development():
    def process(self,X_Train_res,Y_train_res,X_test_data,Y_test):
        try:
            self.kn = KNeighborsClassifier(n_neighbors=5)
            self.gn = GaussianNB()
            self.lr = LogisticRegression()
            self.dt = DecisionTreeClassifier(criterion='entropy')
            self.rf = RandomForestClassifier()

            def Knn(X_Train_res, Y_train_res, X_test_data, Y_test):
                self.kn.fit(X_Train_res, Y_train_res)
                print(f'the Training accuracy:{self.kn.score(X_Train_res, Y_train_res)}')
                print(f'the Testing accuracy:{self.kn.score(X_test_data, Y_test)}')
                y_test_predict = self.kn.predict(X_test_data)
                print(f'confusion matrix:{confusion_matrix(Y_test, y_test_predict)}')
                print(f'classification_report:{classification_report(Y_test, y_test_predict)}')

            def Naive(X_Train_res, Y_train_res, X_test_data, Y_test):
                self.gn.fit(X_Train_res, Y_train_res)
                print(f'the Training accuracy:{self.gn.score(X_Train_res, Y_train_res)}')
                print(f'the Testing accuracy:{self.gn.score(X_test_data, Y_test)}')
                y_test_predict = self.gn.predict(X_test_data)
                print(f'confusion matrix:{confusion_matrix(Y_test, y_test_predict)}')
                print(f'classification_report:{classification_report(Y_test, y_test_predict)}')

            def LR(X_Train_res, Y_train_res, X_test_data, Y_test):
                self.lr = LogisticRegression()
                self.lr.fit(X_Train_res, Y_train_res)
                print(f'the Training accuracy:{self.lr.score(X_Train_res, Y_train_res)}')
                print(f'the Testing accuracy:{self.lr.score(X_test_data, Y_test)}')
                y_test_predict = self.lr.predict(X_test_data)
                print(f'confusion matrix:{confusion_matrix(Y_test, y_test_predict)}')
                print(f'classification_report:{classification_report(Y_test, y_test_predict)}')

            def DT(X_Train_res, Y_train_res, X_test_data, Y_test):
                self.dt = DecisionTreeClassifier(criterion='entropy')
                self.dt.fit(X_Train_res, Y_train_res)
                print(f'the Training accuracy:{self.dt.score(X_Train_res, Y_train_res)}')
                print(f'the Testing accuracy:{self.dt.score(X_test_data, Y_test)}')
                y_test_predict = self.dt.predict(X_test_data)
                print(f'confusion matrix:{confusion_matrix(Y_test, y_test_predict)}')
                print(f'classification_report:{classification_report(Y_test, y_test_predict)}')

            def RF(X_Train_res, Y_train_res, X_test_data, Y_test):
                self.rf = RandomForestClassifier()
                self.rf.fit(X_Train_res, Y_train_res)
                print(f'the Training accuracy:{self.rf.score(X_Train_res, Y_train_res)}')
                print(f'the Testing accuracy:{self.rf.score(X_test_data, Y_test)}')
                y_test_predict = self.rf.predict(X_test_data)
                print(f'confusion matrix:{confusion_matrix(Y_test, y_test_predict)}')
                print(f'classification_report:{classification_report(Y_test, y_test_predict)}')

            def tech(X_Train_res, Y_train_res, X_test_data, Y_test):
                print("    KNN Algorithmn ")
                Knn(X_Train_res, Y_train_res, X_test_data, Y_test)
                print("    NavieBaye's  ")
                Naive(X_Train_res, Y_train_res, X_test_data, Y_test)
                print("    LogisticRegression ")
                LR(X_Train_res, Y_train_res, X_test_data, Y_test)
                print("    DecisionTreeClassifier ")
                DT(X_Train_res, Y_train_res, X_test_data, Y_test)
                print("    RandomForestClassifier ")
                RF(X_Train_res, Y_train_res, X_test_data, Y_test)

            tech(X_Train_res, Y_train_res, X_test_data, Y_test)


        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')
    def best_curve(self,X_Train_res, Y_train_res, X_test_data, Y_test):
        try:
            # Selecting the Best AUC and ROC OF Models
            # Model fit the Training Data
            self.kn.fit(X_Train_res, Y_train_res)
            self.gn.fit(X_Train_res, Y_train_res)
            self.lr.fit(X_Train_res, Y_train_res)
            self.dt.fit(X_Train_res, Y_train_res)
            self.rf.fit(X_Train_res, Y_train_res)

            # Predict the Testing data
            ytest_predict_kn=self.kn.predict_proba( X_test_data)[:,1]
            ytest_predict_gn=self.gn.predict_proba(X_test_data)[:,1]
            ytest_predict_lr=self.lr.predict_proba(X_test_data)[:,1]
            ytest_predict_dt = self.dt.predict_proba(X_test_data)[:, 1]
            ytest_predict_rf = self.rf.predict_proba(X_test_data)[:, 1]

            # finding the best AUC AND ROC of Models[false positive rate -x and true positive rate-y
            fprk,tpk,threshold=roc_curve(Y_test,ytest_predict_kn)
            fprg,tprg,threshold=roc_curve(Y_test,ytest_predict_gn)
            fprl,tprl,threshold=roc_curve(Y_test,ytest_predict_lr)
            fprd,tprd,threshold=roc_curve(Y_test,ytest_predict_dt)
            fprr,tprr,threshold=roc_curve(Y_test,ytest_predict_rf)

            # lets plot them roc curve for all models
            plt.figure(figsize=(6,6))
            plt.plot([0,1],[0,1],'k--')
            plt.plot(fprk,tpk,label='KNN')
            plt.plot(fprg,tprg, label='Navie Bayes')
            plt.plot(fprl,tprl, label='LR')
            plt.plot(fprd,tprd, label='Decision Tree')
            plt.plot(fprr,tprr, label='Random Forest')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend(loc=0)
            plt.show()
            #Logistic Regression nears the True Positive Rates in roc curve
             # lets check the transcation is fraud or not
            self.lr.predict([[-1.112357,0.610467,0.635211,1.040790,-0.835557,-0.543033,-1.265915,-0.397598,-0.543280,-0.430953,-0.478009,-0.862811,-0.184556,1.041463]])
            if self.lr.predict([[-1.112357,0.610467,0.635211,1.040790,-0.835557,-0.543033,-1.265915,-0.397598,-0.543280,-0.430953,-0.478009,-0.862811,-0.184556,1.041463]])[0]==0:
                logging.info('Fraud Transcation')
            else:
                logging.info('Good Transcation')

        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')
class Credit(Model_development):
    def __init__(self):
        self.df=pd.read_csv('T:/Top Mentor/GitHub/Assignments/creditcard.csv')
        self.info=self.df.info# get the information of dataset
        self.nunique=self.df.nunique()#checking duplicates
        self.trans = OneHotEncoder(drop='first')
        self.od = OrdinalEncoder()
        self.le = LabelEncoder()
        self.sc=StandardScaler()
        self.var = VarianceThreshold(threshold=0.0)
        self.var_1=VarianceThreshold(threshold=1.0)
        self.l1 = DropCorrelatedFeatures(method='pearson', threshold=0.8)
        self.sm=SMOTE(random_state=2)
    def Split_data(self):
        try:
            # split the data independent variable and dependent variable
            x=self.df.iloc[:,:-1]
            y=self.df.iloc[:,-1]
            X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=11)
            logging.info(f'X_train:{len(X_train)} and y_train:{len(y_train)}')# checking their len of train and test data
            logging.info(f'X_test:{len(X_test)} and y_test:{len(y_test)}')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')

    def Dataprocessing(self,Training_data,Testing_data):
        try:
           # ----- TRAINING DATA FOR REPLACE NULL------------
            logging.info(Training_data.isna().sum())  # checking the null values and most of the null value is 2 lets check it 'NAN'
            logging.info(Training_data['NPA Status']=='NaN')
            logging.info(Training_data[Training_data['NPA Status'].isnull()].index)# take their index and remove them
            Training_data=Training_data.drop([150000, 150001],axis=0)
            logging.info(f'NULL value after removed 2 observation:{Training_data.isnull().sum()}')
            # MonthlyIncome and MonthlyIncome.1 have same values lets check its repeated value or not by std()
            logging.info(f"MonthlyIncome:{Training_data['MonthlyIncome'].std()}")
            logging.info(f"MonthlyIncome:{Training_data['MonthlyIncome.1'].std()}")
            # both has same value lets remove one of MonthlyIncome.1
            Training_data=Training_data.drop(['MonthlyIncome.1'],axis=1)
            logging.info(Training_data.dtypes)# checking the datatypes
            # NumberOfDependents values are float but dtypes says object lets convert it
            Training_data['NumberOfDependents']=pd.to_numeric(Training_data['NumberOfDependents'])
            # Replace the Null Values with Best Technique Mean,Mode,Median
            mean_1=Training_data['MonthlyIncome'].mean()
            median_1=Training_data['MonthlyIncome'].median()
            mode_1=Training_data['MonthlyIncome'].mode()[0]
            logging.info(f'Monthly income of mean-{mean_1},median:{median_1},mode:{mode_1}')
            def null_replace(dfs,var,value,name):
                dfs[var+name+'_replaced']=dfs[var].fillna(value)
            null_replace(Training_data, 'MonthlyIncome', mean_1, '_mean')
            null_replace(Training_data, 'MonthlyIncome', median_1, 'median')
            null_replace(Training_data, 'MonthlyIncome', mode_1, 'mode')

            def plotting(dfs, var, var1, var2, var3):
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(1, 1, 1)
                dfs[var].plot(color='r', ax=ax, kind='kde', legend='var')
                dfs[var1].plot(color='b', ax=ax, kind='kde', legend='var1')
                dfs[var2].plot(color='y', ax=ax, kind='kde', legend='var2')
                dfs[var3].plot(color='m', ax=ax, kind='kde', legend='var3')
                plt.savefig("plot.png")
            plotting(Training_data,'MonthlyIncome','MonthlyIncome_mean_replaced','MonthlyIncomemedian_replaced','MonthlyIncomemode_replaced')
            plt.show()
            # Random Sample Technique
            Training_data['MonthlyIncome_random']=Training_data['MonthlyIncome'].copy()
            s=Training_data['MonthlyIncome'].dropna().sample(Training_data['MonthlyIncome'].isnull().sum(),random_state=11)
            s.index=Training_data[Training_data['MonthlyIncome'].isnull()].index
            Training_data.loc[Training_data['MonthlyIncome'].isnull(),'MonthlyIncome_random']=s
            # plotting Random Sample Imputation
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            Training_data['MonthlyIncome'].plot(color='r', ax=ax, kind='kde', legend='MonthlyIncome')
            Training_data['MonthlyIncome_random'].plot(color='b', ax=ax, kind='kde', legend='MonthlyIncome_random')
            plt.savefig("plot_1.png")
            plt.show()
            # after plotting lets see std ()
            logging.info(f"MonthlyIncome:{Training_data['MonthlyIncome'].std()}")
            logging.info(f"MonthlyIncome_mean:{Training_data['MonthlyIncome_mean_replaced'].std()}")
            logging.info(f"MonthlyIncome_median:{Training_data['MonthlyIncomemedian_replaced'].std()}")
            logging.info(f"MonthlyIncome_mode:{Training_data['MonthlyIncomemode_replaced'].std()}")
            logging.info(f"MonthlyIncome_random:{Training_data['MonthlyIncome_random'].std()}")
            # Mode and Random seems very close to monthly income points
            Training_data = Training_data.drop(['MonthlyIncome','MonthlyIncome_mean_replaced','MonthlyIncomemedian_replaced','MonthlyIncome_random'], axis=1)
            mean_num = Training_data['NumberOfDependents'].mean()
            median_num = Training_data['NumberOfDependents'].median()
            mode_num = Training_data['NumberOfDependents'].mode()[0]
            logging.info(f'NumberOfDependents of mean:-{mean_num},median:{median_num},mode:{mode_num}')
            null_replace(Training_data, 'NumberOfDependents', mean_num,'_mean')
            null_replace(Training_data, 'NumberOfDependents', median_num,'median')
            null_replace(Training_data, 'NumberOfDependents', mode_num,'mode')
            logging.info(Training_data.columns)
            # Let Plot them No of dependents
            def plotting(dfs, var, var1, var2, var3):
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(1, 1, 1)
                dfs[var].plot(color='r', ax=ax, kind='kde', legend='var')
                dfs[var1].plot(color='b', ax=ax, kind='kde', legend='var1')
                dfs[var2].plot(color='y', ax=ax, kind='kde', legend='var2')
                dfs[var3].plot(color='m', ax=ax, kind='kde', legend='var3')
                plt.savefig("plot_2.png")
            plotting(Training_data, 'NumberOfDependents','NumberOfDependents_mean_replaced','NumberOfDependentsmedian_replaced',
                     'NumberOfDependentsmode_replaced')
            plt.show()
            # after plotting lets see std ()
            logging.info(f"NumberOfDependents:{Training_data['NumberOfDependents'].std()}")
            logging.info(f"NumberOfDependents_mean:{Training_data['NumberOfDependents_mean_replaced'].std()}")
            logging.info(f"NumberOfDependents_median:{Training_data['NumberOfDependentsmedian_replaced'].std()}")
            logging.info(f"NumberOfDependents_mode:{Training_data['NumberOfDependentsmode_replaced'].std()}")
            # lets take Mean holds the value Median and Mode has same value
            Training_data = Training_data.drop(['NumberOfDependents', 'NumberOfDependentsmedian_replaced', 'NumberOfDependentsmode_replaced'], axis=1)
            logging.info(f'After Replaced NULL value :{Training_data.isnull().sum()}')

            # # ----- TESTNG DATA FOR REPLACE NULL------------
            # Apply the same Technic in Testing DATA
            logging.info(Testing_data.isna().sum())  # checking the null values and most of the null value is 2 lets check it 'NAN'
            # both has same value lets remove one of MonthlyIncome.1
            Testing_data = Testing_data.drop(['MonthlyIncome.1'], axis=1)
            # NumberOfDependents values are float but dtypes says object lets convert it
            Testing_data['NumberOfDependents'] = pd.to_numeric(Testing_data['NumberOfDependents'])
            logging.info(Testing_data.dtypes)  # checking the datatypes
            # Replace the Null Value form Training data
            Testing_data['MonthlyIncome'] = Testing_data['MonthlyIncome'].fillna(mode_1)
            Testing_data['NumberOfDependents'] = Testing_data['NumberOfDependents'].fillna(mean_num)
            # After replacing Null values check them
            logging.info(Testing_data.isna().sum())

                         # -------Explortory Data Analysis-------
            numeric_feature=[feature for feature in Training_data.columns if Training_data[feature].dtypes!='O']
            logging.info(numeric_feature)
            categorical_feature = [feature for feature in Training_data.columns if Training_data[feature].dtypes == 'O']
            logging.info(categorical_feature)
            # Univarte Analysis for Numeric columns-around 30 to 60 having a lot of transcation in card
            plt.figure(figsize=(5, 5))
            plt.hist(Training_data['age'], bins=10)
            plt.title("numerical univariate analysis")
            plt.show()
            logging.info(Training_data['Gender'].unique())
            # univariate analysis categorical column
            fur_name = Training_data['Gender'].value_counts().index
            fur_val = Training_data['Gender'].value_counts().values
            plt.pie(fur_val, labels=fur_name, autopct='%1.2f%%')# Male has borrowed 61.58% for card transcation
            plt.title("univariate analysis Categorical")
            plt.show()
            # Bivarte Analysis
            plt.figure(figsize=(5, 5))  # AGE OF BORROWERS IN YEARS and their montly income
            sns.scatterplot(x=Training_data['age'], y=Training_data['MonthlyIncomemode_replaced'])
            plt.show()
            # checking the relationship between monthly income,NumberOfOpenCreditLinesAndLoans with dependent variable Good_Bad
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            sns.boxplot(x='MonthlyIncomemode_replaced', y='Good_Bad', data=Training_data)
            plt.subplot(1, 2, 2)
            sns.violinplot(x='NumberOfOpenCreditLinesAndLoans', y='Good_Bad', data=Training_data, palette='rainbow')
            plt.show()
                                 # ----- TRAINING DATA FOR Normal Distribution ,OUTLIERS ------------
            # numeric columns for Normal Distribution
            Training_data_numeric = Training_data.select_dtypes(exclude='object')
            logging.info(Training_data_numeric.columns)
            fig, ax = plt.subplots(6, 2, figsize=(23, 19))
            for i, subplot in zip(Training_data_numeric.columns, ax.flatten()):
                sns.distplot(Training_data[i], color='b', ax=subplot)
            plt.show()
            # Normal Distribution doesn't look good and lets see outliers is present or not
            fig,ax=plt.subplots(6,2,figsize=(20,21))
            for i,subplot in zip(Training_data_numeric.columns,ax.flatten()):
                sns.boxplot(Training_data_numeric[i],orient='h',color='g',ax=subplot)
            plt.show()
            #BOTH normal distribution and boxplot of training data kde doesn't look good lets check with variable transformation Technique
            Training_data_numeric['NumberOfDependents_mean_replaced_yeo'],alpha=stats.yeojohnson(Training_data_numeric['NumberOfDependents_mean_replaced'])
            sns.boxplot(data=Training_data_numeric,x='NumberOfDependents_mean_replaced_yeo', orient='h', color='g')
            plt.show()
            Training_data_numeric=Training_data_numeric.drop(['NumberOfDependents_mean_replaced_yeo'],axis=1)
            #  yeo johnson gives good results so Applying yeo johnson technique for all training numeric data
            def plotts(df,var):
                plt.figure(figsize=(5,5))
                plt.subplot(1,3,1)
                df[var+'_yeo'].hist(bins=5,grid=False)
                plt.title('hist')
                plt.subplot(1, 3, 2)
                sns.boxplot(df[var+'_yeo'],color='b',orient='h')
                plt.title('boxplot')
                plt.subplot(1, 3, 3)
                stats.probplot(df[var+'_yeo'],plot=plt)
                plt.show()
            for i in Training_data_numeric.columns:
                Training_data_numeric[i+'_yeo'], alpha = stats.yeojohnson(Training_data_numeric[i])
                plotts(Training_data_numeric, i)
            logging.info(Training_data_numeric.columns)
            # Even those YeoJOhnson technique gives good results but few features has no unique value in data so lets remove them and also 0utlier is there ..
            # k[0],k[3],k[6],k[8]
            logging.info(f"NPA Status_yeo:,{Training_data_numeric['NPA Status_yeo'].unique()}")
            logging.info(f"NumberOfTimes90DaysLate_yeo:,{Training_data_numeric['NumberOfTimes90DaysLate_yeo'].unique()}")
            logging.info(f"NumberOfTime60-89DaysPastDueNotWorse_yeo:,{Training_data_numeric['NumberOfTime60-89DaysPastDueNotWorse_yeo'].unique()}")
            logging.info(f"NumberOfTime30-59DaysPastDueNotWorse_yeo::,{Training_data_numeric['NumberOfTime30-59DaysPastDueNotWorse_yeo'].unique()}")
            # only NumberOfTime30-59DaysPastDueNotWorse_yeo shows lot of unique and remove others
            # yeo johnson gives good results but it shows some feature with Outliers
            Training_datas=Training_data_numeric[['RevolvingUtilizationOfUnsecuredLines_yeo', 'age_yeo','NumberOfTime30-59DaysPastDueNotWorse_yeo', 'DebtRatio_yeo','NumberOfOpenCreditLinesAndLoans_yeo','NumberRealEstateLoansOrLines_yeo','MonthlyIncomemode_replaced_yeo','NumberOfDependents_mean_replaced_yeo']]
            # Taking only features with applied yeo johnson features

                                            # ---- OUTLIERS Training Data--------
            def quan(dfs, varn):
               quantile_1 = dfs[varn].quantile(0.25)
               quantile_3 = dfs[varn].quantile(0.75)
               IQR = quantile_3 - quantile_1
               upper_limit = quantile_3 + 1.5 * IQR
               lower_limit = quantile_1 - 1.5 * IQR
               return upper_limit,lower_limit
            for i in Training_datas.columns:
               upper, lower =quan(Training_datas,i)
               Training_datas[i]=np.where(Training_datas[i]>upper,upper,np.where(Training_datas[i]<lower,lower,Training_datas[i]))

            #lets plot them yeojoheson features and after applying outliers lets check it
            fig, ax = plt.subplots(4, 2, figsize=(23, 19))
            for i, subplot in zip(Training_datas.columns, ax.flatten()):
               sns.boxplot(data=Training_datas,x=i, ax=subplot, color='b', orient='h')
               plt.title('After Applying yeo johnson and Qutliers for Training Data')
            plt.show()
                        # ----- TESING DATA FOR Variable ,OUTLIERS ------------
               # numeric columns for TESING DATA
            Testing_data_numeric = Testing_data.select_dtypes(exclude='object')
            logging.info(len(Testing_data_numeric.columns))
            for i in Testing_data_numeric.columns:
               Testing_data_numeric[i+'yeo'],alpha=stats.yeojohnson(Testing_data_numeric[i])
            # Removing FEATURE which has no unique
            Testing_datas=Testing_data_numeric[['RevolvingUtilizationOfUnsecuredLinesyeo', 'ageyeo','NumberOfTime30-59DaysPastDueNotWorseyeo', 'DebtRatioyeo','NumberOfOpenCreditLinesAndLoansyeo','NumberRealEstateLoansOrLinesyeo', 'MonthlyIncomeyeo','NumberOfDependentsyeo']]
            logging.info(Testing_datas.columns)

            def Iqr(Testing_datas, var):
               quantile_1 = Testing_datas[var].quantile(0.25)
               quantile_3 = Testing_datas[var].quantile(0.75)
               IQR = quantile_3 - quantile_1
               upper_limit = quantile_3 + (1.5 * IQR)
               lower_limit = quantile_1 - (1.5 * IQR)
               return upper_limit, lower_limit
            for i in Testing_datas.columns:
               upper, lower = Iqr(Testing_datas, i)
               Testing_datas[i] = np.where(Testing_datas[i] > upper, upper,
                                         np.where(Testing_datas[i] < lower, lower, Testing_datas[i]))
            # lets plot them yeojoheson features and after applying outliers lets check it
            fig, ax = plt.subplots(4, 2, figsize=(23, 19))
            for i, subplot in zip(Testing_datas.columns, ax.flatten()):
               sns.boxplot(data=Testing_datas, x=i, ax=subplot, color='g')
               plt.title("yeojoheson features and after applying outliers for Testing Data")
            plt.show()

                       # Converting Categorical to Numeric [TRAINING Data]
            Train_cat_cols=Training_data.select_dtypes(include='object')
            # ONEHOTENCODER FOR Training DATA
            self.trans.fit(Train_cat_cols[['Gender', 'Rented_OwnHouse', 'Region']])
            cat =self.trans.transform(Train_cat_cols[['Gender', 'Rented_OwnHouse', 'Region']])
            cat = cat.toarray()
            cat = pd.DataFrame(cat, index=Train_cat_cols.index)
            cat.columns =self.trans.get_feature_names_out()
            Train_cat_cols = pd.concat([Train_cat_cols, cat], axis=1)
            # OrdinalEncoder FOR Training DATA
            od_data = self.od.fit_transform(Train_cat_cols[['Occupation', 'Education']])
            od_data = pd.DataFrame(od_data, index=Train_cat_cols.index)
            od_data.columns = ['Occupations', 'Educations']
            Train_cat_cols = pd.concat([Train_cat_cols, od_data], axis=1)
            # LABELENCODER FOR DEPENDENT Training_data
            label_code = self.le.fit_transform(Train_cat_cols['Good_Bad'])
            label_code = pd.DataFrame(label_code, index=Train_cat_cols.index)
            label_code.columns = ['Good_Bad_changed']
            Train_cat_cols = pd.concat([Train_cat_cols, label_code], axis=1)
            Train_cat_cols = Train_cat_cols.drop(['Gender', 'Region', 'Rented_OwnHouse', 'Occupation', 'Education','Good_Bad'], axis=1)
            # finally converted cate to num features
            Final_training_data = pd.concat([Training_datas, Train_cat_cols], axis=1)
            logging.info(f" Final_training_data:{Final_training_data.columns}")

                  # -----  Converting Categorical to Numeric [TESTING Data]
            Testing_data_cate = Testing_data.select_dtypes(include='object')
            # ONEHOTENCODER FOR TESTING DATA
            oh =self.trans.transform(Testing_data_cate[['Gender', 'Rented_OwnHouse', 'Region']])
            oh = oh.toarray()
            oh = pd.DataFrame(oh, index=Testing_data_cate.index)
            oh.columns =self.trans.get_feature_names_out()
            Testing_data_cate = pd.concat([Testing_data_cate, oh], axis=1)
            # ORDINALENCODER FOR TESTING DATA
            od_data =self.od.transform(Testing_data_cate[['Occupation', 'Education']])
            od_data = pd.DataFrame(od_data, index=Testing_data_cate.index)
            od_data.columns = ['Occupations', 'Educations']
            Testing_data_cate = pd.concat([Testing_data_cate, od_data], axis=1)
            # LABELENCODER FOR DEPENDENT testing_data
            le_data =self.le.transform(Testing_data_cate['Good_Bad'])
            le_data = pd.DataFrame(le_data, index=Testing_data_cate.index)
            le_data.columns = ['Good_Bad_changed']
            Testing_data_cate = pd.concat([Testing_data_cate, le_data], axis=1)
            Testing_data_cate = Testing_data_cate.drop(['Gender', 'Region', 'Rented_OwnHouse', 'Occupation', 'Education',
                                                    'Good_Bad'], axis=1)
            Final_testing_data = pd.concat([Testing_datas,Testing_data_cate],axis=1)
            logging.info(f"Final_testing_data:{Final_testing_data.columns}")
            return Final_training_data,Final_testing_data
        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')


    def Featurescale(self,Final_training_data, Final_testing_data):
        try:
            X_train = Final_training_data.iloc[:, :-1]
            Y_train = Final_training_data.iloc[:, -1]
            X_test = Final_testing_data.iloc[:, :-1]
            Y_test = Final_testing_data.iloc[:, -1]
            # scaling down the values will gives good results
            X_train_data =self.sc.fit_transform(X_train)
            X_test_data =self.sc.transform(X_test)
            X_train_data=pd.DataFrame(X_train_data,columns=X_train.columns)
            X_test_data=pd.DataFrame(X_test_data,columns=X_test.columns)
                    # After feature Scaling plot them by boxplot
            fig, ax = plt.subplots(4, 2, figsize=(23, 19))
            for i, subplot in zip(X_train_data.columns, ax.flatten()):
                sns.boxplot(data=X_train_data, x=i, ax=subplot, color='b')
                plt.title('After feature Scaling plot them by boxplot')
            plt.show()
                     # After feature Scaling plot them by boxplot
            fig, ax = plt.subplots(4, 2, figsize=(23, 19))
            for i, subplot in zip(X_test_data.columns, ax.flatten()):
                sns.boxplot(data=X_test_data, x=i, ax=subplot, color='r')
                plt.title('After feature Scaling plot them by boxplot')
            plt.show()

                     # ------ Feature Selection---------
            #1.Variance Threshold
            self.var.fit(X_train_data)
            c1=X_train_data.columns[self.var.get_support()]# needed column which has greater than 0
            c2=X_train_data.columns[~self.var.get_support()]# unwanted columns
            logging.info(f"Unwanted Columns:{c2}")
            # Lets remove them after the value less than variance threshold
            X_train_data=X_train_data.drop(['NumberOfTime30-59DaysPastDueNotWorse_yeo'],axis=1)
            X_test_data =X_test_data.drop(['NumberOfTime30-59DaysPastDueNotWorseyeo'], axis=1)
            # 1.1Variance Threshold
            self.var_1.fit(X_train_data)
            d1=X_train_data.columns[self.var_1.get_support()]# needed column
            d2=X_train_data.columns[~self.var_1.get_support()]# unwanted column
            logging.info(d2)
            # 2.Feature Selection using Correlation
            correlat=self.l1.fit(X_train_data[['RevolvingUtilizationOfUnsecuredLines_yeo', 'age_yeo', 'DebtRatio_yeo','NumberOfOpenCreditLinesAndLoans_yeo','NumberRealEstateLoansOrLines_yeo', 'MonthlyIncomemode_replaced_yeo','NumberOfDependents_mean_replaced_yeo']])
            logging.info(correlat.features_to_drop_)
            #3.1.Hypothesis testing for Numeric
            #Lets do Correlation in Pearson Technique and p_value = 0.05 if it is greater nullhypothesis is rejected and if lessernull hypothesis is selected
            training_data_numeric = X_train_data[['RevolvingUtilizationOfUnsecuredLines_yeo', 'age_yeo', 'DebtRatio_yeo','NumberOfOpenCreditLinesAndLoans_yeo','NumberRealEstateLoansOrLines_yeo', 'MonthlyIncomemode_replaced_yeo','NumberOfDependents_mean_replaced_yeo']]
            corr = []
            for i in training_data_numeric.columns:
                cor = pearsonr(training_data_numeric[i], Y_train)
                corr.append(cor)
            corr = np.array(corr)# convert to array
            logging.info(f'corr:{corr}')
            p_value_num = pd.Series(corr[:, 1], index=training_data_numeric.columns)
            logging.info(f'p_value_num:{p_value_num}')
            p_value_num = p_value_num.sort_values(ascending=True)
            #lets plot them
            p_value_num.plot.bar()
            plt.show()
            X_train_data = X_train_data.drop(['DebtRatio_yeo'], axis=1)
            X_test_data = X_test_data.drop(['DebtRatioyeo'], axis=1)
            # 3.2.Hypothesis testing for Categorical
            Train_cat_col = X_train_data[['Gender_Male',
                                          'Rented_OwnHouse_Rented', 'Region_East', 'Region_North', 'Region_South',
                                          'Region_West', 'Occupations', 'Educations']]
            logging.info(Train_cat_col.head())#chi2 doesn't applicable here due to negative values
            # Lets check the data is balanced or imbalanced[dependent variable]
            logging.info(f'the value of dependent variable:{sum(Y_train==1)}')
            logging.info(f'the value of dependent variable:{sum(Y_train==0)}')
            X_Train_res,Y_train_res=self.sm.fit_resample(X_train_data,Y_train)
            logging.info(f'the value of dependent variable:{sum(Y_train_res == 1)}')
            logging.info(f'the value of dependent variable:{sum(Y_train_res == 0)}')
            return X_Train_res,Y_train_res,X_test_data,Y_test

        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')

    def check_data(self):
        try:
            logging.info(self.info)# Basic Information for Datasets
            logging.info(self.nunique)#checking duplicates
            logging.info(f'Number of obeservation and features:{self.df.shape}')
            X_train, X_test, y_train, y_test = self.Split_data()
            Training_data = pd.concat([X_train, y_train], axis=1)  # combining the training data
            Testing_data = pd.concat([X_test, y_test], axis=1)  # combine the X_test and y_test
            Final_training_data, Final_testing_data = obj.Dataprocessing(Training_data, Testing_data)
            X_Train_res, Y_train_res, X_test_data, Y_test = obj.Featurescale(Final_training_data, Final_testing_data)
            obj.process(X_Train_res, Y_train_res, X_test_data, Y_test )
            self.best_curve(X_Train_res, Y_train_res, X_test_data, Y_test)

        except Exception as e:
            logging.error(f'error in main:{e.__str__()}')

if __name__ == '__main__':
    try:
        obj=Credit()
        obj.check_data()

    except Exception as e:
        logging.error(f'error in main:{e.__str__()}')