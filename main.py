

import csv
from tkinter import E
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import random
# Item filtering
#tfd id on categories
CONST_LIMIT_USER_EXEC=200
CONST_MAX_NUMBER_MOVIES = 25000
CONST_MAX_NUMBER_LINE_READ = 200000
TRAIN_TEST_DIVIDER_RATE = 0.75
SIMILARITY_MINIMUM_ITEM_BASED = 0.00005
SIMILARITY_MINIMUM_USER_BASED = 0.00005
RATIO_MODELS = 0.3

def read_Movies(moviefilename):
    dMovRealId = {}
    mMovCattxt = []
    mMovtitletxt = []
    idInFile=0
    lastMovieId = 0
    dMovtitletxt ={}
    i = 0
    with open(moviefilename, 'r') as file:
        csvreader = csv.reader(file)
        for e in csvreader:
            if i!=0  and int(e[0]) < CONST_MAX_NUMBER_MOVIES :
                idInFile+=1
                mMovCattxt.append(e[2].replace("|"," "))
                pos =e[1].find("(19")
                posforrep = 0
                if pos != -1:
                    posforrep = pos
                elif e[1].find("(20")!=-1:
                    posforrep = e[1].find("(20")
                title = e[1].replace(e[1][posforrep:posforrep+6],"")
                mMovtitletxt.append(title)
                dMovRealId[int(e[0])]=int(idInFile)
                dMovtitletxt[int(e[0])]= title
                lastMovieId = e[0]
            i+=1
    return dMovRealId,mMovCattxt,mMovtitletxt,int(lastMovieId),dMovtitletxt

def get_tf_idf(mMov) :
    vectorizer = TfidfVectorizer(input = mMov, use_idf=True  )
    X = vectorizer.fit_transform(mMov)
    return X

def get_ratings_of_user(ratingsFileName,user):
    dTest= {}
    dTrain= {}
    i = 0
    random.seed(50)
    with open(ratingsFileName, 'r') as file:
        csvreader = csv.reader(file)
        for e in csvreader:
            if i!=0:
                idUser = int(e[0])
                idMovie = int(e[1])
                if user == idUser and idMovie < CONST_MAX_NUMBER_MOVIES:
                    i = random.random()
                    if i >= TRAIN_TEST_DIVIDER_RATE :
                        dTest[int(e[1])]=float(e[2])
                    else: 
                        dTrain[int(e[1])]=float(e[2])
                elif user<idUser: 
                    break
            i=+1
    return dTrain,dTest

def getAllUsers(ratings_file_name):
    lUsers = []
    i = 0
    lastuser = 0
    with open(ratings_file_name, 'r') as file:
        csvreader = csv.reader(file)
        for e in csvreader:
            if i!=0:
                lUsers.append(int(e[0]))
                lastuser = e[0]
            i+=1
            if i == CONST_MAX_NUMBER_LINE_READ:
                break
    return set(lUsers),int(lastuser)

def getAllUsersRatings(ratings_file_name,dAllUsers):
    dUsersRatings = {}
    for user in dAllUsers: 
        dUsersRatings[int(user)] = get_ratings_of_user(ratings_file_name,int(user))
    return dUsersRatings


def score_item_based_for_a_user(tfidfcat,tfidftitle,NonRatedMovies,RatedMovies,dMovies):
    dScoreMovie = {}
    s1=0
    s2=0
    # Param ratings pour un utilisateur
    for nrmRealInex, nrmFileIndex in NonRatedMovies.items(): 
        dividercat = 0
        dividertitle = 0
        for mov, rat in RatedMovies.items():
            if rat != 0 :
                distcat = float(cosine_similarity(tfidfcat[dMovies[mov]-1], tfidfcat[nrmFileIndex-1], dense_output=True))
                disttitle = float(cosine_similarity(tfidftitle[dMovies[mov]-1], tfidftitle[nrmFileIndex-1], dense_output=True))
                if distcat>SIMILARITY_MINIMUM_ITEM_BASED:
                    s1 = s1 +  (rat *  distcat)
                    s2 = s2 +  (rat * disttitle)
                    dividercat +=distcat
                    dividertitle += disttitle 

            if dividertitle == 0 :
                dividertitle = 1
            if dividercat == 0:
                dividercat = 1
                
        dScoreMovie[nrmRealInex]=(float(s1)/dividercat)*0.95+(float(s2)/dividertitle)*0.05
        s1=0
        s2=0
    return dScoreMovie

def convert_dict_of_dict_to_list_of_list(dUsersRatings,real_nbr_user,real_nbrMovies):
    matrix_ratings = [ [0.0] *(real_nbrMovies+1) ] * (real_nbr_user+1)
    for u in dUsersRatings.keys():
        templ= [0]* (real_nbrMovies+1)
        dict_r = dUsersRatings[u]
        for indx,v in dict_r.items():
            templ[indx] = v
        matrix_ratings[u] = templ
    return matrix_ratings


def notRatedByUser(dTest,dMovies):
    dnotRated = {}
    for key, value in dMovies.items():
        if key in dTest:
            dnotRated[key] =  value  
    return dnotRated


def score_collaborative_for_a_user(dUsersRatings, dUserRatings,dnotRated,dScoreMovieItemBased,real_nbr_user,real_nbrMovies):
    dScoreMovie={}
    matrix_otherUser = convert_dict_of_dict_to_list_of_list(dUsersRatings,real_nbr_user,real_nbrMovies)
    # Convert dUserRating ( dictionary of dictionaries countaining the raatings for each user ) into a list of lists 
    list_user_ratings = [0] * (real_nbrMovies + 1)
    for indx,v in dUserRatings.items():
        list_user_ratings[int(indx)] = v
    prototypeVecteurNull = [0] * (real_nbrMovies + 1)

    for nrm in dnotRated.keys():
        s=0
        divider = 0
        for rating_user in matrix_otherUser:
            if rating_user != prototypeVecteurNull:
                if rating_user[nrm] != 0:
                    dist = np.corrcoef(rating_user, list_user_ratings)[1][0] 
                    if dist > SIMILARITY_MINIMUM_USER_BASED: 
                        s  = s + ( rating_user[nrm] *dist ) 
                            
                        divider+=dist
        if divider == 0:
            divider = 1
        dScoreMovie[nrm]= s/divider
    return dScoreMovie

def multiModelsFusion(dmodel1,dmodel2):
    dfinalModel={}
    for k,v in dmodel1.items():
        val = v*RATIO_MODELS+dmodel2[k]*(1-RATIO_MODELS)
        dfinalModel[k]=val
    return dfinalModel

def eval_with_cosinus(dresults,dtestuser):
    size = len(dresults.keys())
    vec1 = [0]*size
    vec2 = [0]*size
    i = 0
    for k,v in dresults.items():
        vec1[i]=v
        vec2[i]=dtestuser[k]
        i+=1

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(vec1)):
        x = vec1[i]; y = vec2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def eval_for_prediction(dresults,dtestuser):
    i=0
    acc=0
    err=0
    for k,v in dresults.items():
        v1=v
        v2=dtestuser[k] 
        delta = math.sqrt((v2-v1)**2)
        if delta <= 1 :
            acc+=1
        if delta > 1:
            err+=1
        i+=1
    if i != 0:
        accuracy = acc/i
    else : 
        accuracy = -1
    return accuracy

def eval_for_recommandation(dresults,dtestuser):
    i=0
    acc=0
    err=0
    for k,v in dresults.items():
        v1=v
        v2=dtestuser[k] 
        delta = v2-v1
        if delta <= 1 :
            acc+=1
        if delta < -1:
            err+=1
        i+=1
    if i != 0:
        recall = acc/i
    else : 
        recall = -1
    return recall,(acc/(acc+err))

def sorted_movies(dfinalModel,dMoviesTitle,user,accuracy,recall,precision):
    sorted_values = sorted(dfinalModel.values()) # Sort the values
    sorted_dict = {}
    for i in sorted_values:
        for k in dfinalModel.keys():
            if dfinalModel[k] == i:
                sorted_dict[k] = dfinalModel[k]
    print("-------------------recommandations for user:"+str(user)+"------------------------")
    for e in reversed(sorted_dict.keys()):
        print(str(dMoviesTitle[e])+" "+str(dfinalModel[e]))
    print("------------")
    print("---Prediction : ")
    print("*** Accuracy = " + str(accuracy))
    print("---Recommandation :")
    print("*** recall = " + str(accuracy))
    print("*** precision = "+ str(precision))
    print("-------------------------------------------------")
    

def main() :
    #Initializing 
    indExec = 0
    avgAccuracyPredict=0
    avgRecallRec=0
    avgPrecisionRec=0
    #Getting the movies from the file
    #dMovies : it's a dictionary wich for each movie id as a Key, the Value represents the location of the movie on the file and also on the tfidf table
    #mMovCattxt : a matrix wich for each movies his categories with a string
    #mMovtitletxt : same thing but for the titles 
    #dMovtitletxt : same thing as mMovtitle but its a dictionary, key : a movie's id, value : the title as a string
    #real_nbrMovies : Its the last Movie id wich had been read 
    dMovies,mMovCattxt,mMovtitletxt,real_nbrMovies,dMovtitletxt = read_Movies("movie.csv")
    tfidfcat = get_tf_idf(mMovCattxt)
    tfidftitle = get_tf_idf(mMovtitletxt)
    #Getting a list of all the users under a certain number (  global variable )
    sAllUsers,real_nbr_user = getAllUsers("rating.csv")
    #getting a dictionary of dictionaries countaining the ratings of each users
    dRatings_users = getAllUsersRatings("rating.csv",sAllUsers)

    for user in sAllUsers:
        # Breaking the execution for a certain number of users 
        if CONST_LIMIT_USER_EXEC == indExec :
            break
        indExec+=1
        dtrainuser, dtestuser = dRatings_users[user]
        #Dictionary for the movies that a user didn't rated, with the same format as dMovies
        dnotRated = notRatedByUser(dtestuser,dMovies)
        #Item Based model ( using the titles and the categories )
        dIBmoviesScore = score_item_based_for_a_user(tfidfcat,tfidftitle,dnotRated, dtrainuser,dMovies)
        dtrainRatings_users = {}
        for k,v in dRatings_users.items():
            dtrainRatings_users[k]=v[0]
        #User Based model
        dUBmoviesScore = score_collaborative_for_a_user(dtrainRatings_users, dtrainuser,dtestuser,dIBmoviesScore,real_nbr_user,real_nbrMovies)
        #Fusion of the used models 
        dfinalModel = multiModelsFusion(dIBmoviesScore,dUBmoviesScore)
        #Prediction evalution with accuracy 
        accuracy = eval_for_prediction(dfinalModel,dtestuser)
        #Recommandation evaluation with recall and precision
        recall,precision = eval_for_recommandation(dfinalModel,dtestuser)
        avgAccuracyPredict += accuracy
        avgRecallRec = avgRecallRec + recall
        avgPrecisionRec =  avgPrecisionRec + precision
        #Displaying the recommended movies, with the highest predicted rates on top, and the lowest on the bottom 
        sorted_movies(dfinalModel,dMovtitletxt,user,accuracy,recall,precision)
    print("----------------------Final results---------------------------")
    print("----Prediction " )
    print("* accuracy prediction = " + str(avgAccuracyPredict/indExec))
    print("----Recommandation results :")
    print("* Average recall = " + str(avgRecallRec/indExec))
    print("* Average precision = "+ str(avgPrecisionRec/indExec))
    print("The End")
    
main()
