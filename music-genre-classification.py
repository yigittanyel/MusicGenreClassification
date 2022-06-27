from __future__ import print_function
from turtle import distance
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import sys

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math

from flask_wtf.file import FileField
from flask import Flask, render_template, request, redirect, url_for #flask
from flask_wtf import FlaskForm
from wtforms import SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey' #form işlemleri için oluşturduğum anahtar, gerekli


#file uploading
class UploadFileForm(FlaskForm):
    file = FileField("File")

@app.route("/")
def mainPage():
     return render_template('first.html')

@app.route("/index")
def indexPage():
    form = UploadFileForm()
    return render_template("index.html", form=form)

@app.route("/home")
def musicGenre():
    form = UploadFileForm()
    return render_template("home.html", form=form)

@app.route('/func', methods=['GET', 'POST'])
def muzik_turunu_bul():
    form = UploadFileForm()
    #veri setindeki veriler uygulamaya öğretildikten öğretme algoritmaları silindi.

    #tüm öğretilmiş veriler my.dat dosyasında saklı. my.dat çok önemli o dosyaya hiçbirşey olmamalı


    #kullanıcı arayüzüne yalnızca 'wav' uzantılı dosyalar eklenilmesi gerektiği yazılmalı!

    #özellik vektörleri arasındaki mesafeyi bulma ve komşuları bulma işlevi  

    #wav harici dosya eklendiği zaman hata gösterilmeli. file.filename sonunda .wav ile bitmiyorsa hata gösterilmeli

    def getKomsular(ogretmeSeti, ornek, k):
        mesafeler = []
        for x in range (len(ogretmeSeti)):
            mesafe = distance(ogretmeSeti[x], ornek, k) + distance(ornek, ogretmeSeti[x], k)
            mesafeler.append((ogretmeSeti[x][2], mesafe))

        mesafeler.sort(key=operator.itemgetter(1))
        komsular = []
        for x in range(k):
            komsular.append(mesafeler[x][0])

        return komsular

    # örnek sınıfı tanımla
    def enYakinSinif(komsular):
        classPuani = {}

        for x in range(len(komsular)):
            cevap = komsular[x]
            if cevap in classPuani:
                classPuani[cevap] += 1
            else:
                classPuani[cevap] = 1

        sira = sorted(classPuani.items(), key = operator.itemgetter(1), reverse=True)

        return sira[0][0]

    dataset = []
    def datasetYukle(filename):
        with open('my.dat', 'rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break

    datasetYukle('my.dat')

    def distance(ornek1 , ornek2 , k ):
        uzaklık =0 
        mm1 = ornek1[0] 
        cm1 = ornek1[1]
        mm2 = ornek2[0]
        cm2 = ornek2[1]
        uzaklık = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
        uzaklık+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
        uzaklık+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        uzaklık-= k
        return uzaklık

    # kodu harici örneklerle test etme
    # URL: https://uweb.engr.arizona.edu/~429rns/audiofiles/audiofiles.html

    #classical'ın içine tarkan şarkısı koydum.
    #country içine hiphop(dapoet) koydum.
  
    file = form.file.data #İlk önce dosyayı al 
    
    file.save(os.path.join(app.root_path, file.filename))

    #flaskın kullanıcı dosyalarına ulaşma yetkisi yok, dosyanın full pathine erişim yetkisi yok, yalnızca seçilen dosyaya yetkisi var
    #aynı isimdeki dosyayı kullandığında üzerine mi yazar yoksa hata mı verir?(cevabını bilmiyoruz, üstüne yazıyor olabilir)

    (rate, sig) = wav.read(os.path.join(app.root_path, file.filename))

    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    kovaryans = np.cov(np.matrix.transpose(mfcc_feat))


    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, kovaryans, 11)

    global results

    results = {1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'}
    #dosya yolu tanımlamaya gerek olmaması için müzik türleri elle tanımlandı

    global pred
    pred = enYakinSinif(getKomsular(dataset, feature, 5))
    print("Çalan müziğin türü:", results[pred])

    os.remove(os.path.join(app.root_path, file.filename)) #çalıştırılıp bulunan dosya klasörden siliniyor

    return render_template("index.html", sonuc= results[pred], form=form)

if __name__ == "__main__":
    app.run(debug=True)