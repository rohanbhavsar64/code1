import streamlit as st
import pickle
import pandas as pd
import numpy as np
car=pd.read_csv('cleandata .csv')
#st.write(df)
x=car[['name','company','year','kms_driven','fuel_type']]
y=car['price']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
ohe=OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_, handle_unknown = 'ignore'),['name','company','fuel_type']),
                                    remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(xtrain,ytrain)
y_pred=pipe.predict(xtest)
#st.write(y_pred)
st.title('Car Price Predicator')
#st.write(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(0,4))))
comp=['Hyundai',
      'Mahindra',
      'Ford',
      'Maruti',
      'Skoda',
      'Audi',
      'Toyota',
      'Renault',
      'Honda',
      'Datsun',
      'Mitsubishi',
      'Tata',
      'Volkswagen',
      'Chevrolet',
      'Mini',
      'BMW',
      'Nissan',
      'Hindustan',
      'Fiat',
      'Force',
      'Mercedes',
      'Land',
      'Jaguar',
      'Jeep',
      'Volvo']
yr=['2007', '2006', '2018', '2014', '2015', '2012', '2013', '2016',
    '2010', '2017', '2008', '2011', '2019', '2009', '2005', '2000',
    '2023','2021','2020','2022','2004','2003','2002','2001']
fuel=['Petrol', 'Diesel', 'nan', 'LPG']
naam=['Hyundai Santro Xing XO eRLX Euro III', 'Mahindra Jeep CL550 MDI',
       'Maruti Suzuki Alto 800 Vxi',
       'Hyundai Grand i10 Magna 1.2 Kappa VTVT',
       'Ford EcoSport Titanium 1.5L TDCi', 'Ford Figo', 'Hyundai Eon',
       'Ford EcoSport Ambiente 1.5L TDCi',
       'Maruti Suzuki Alto K10 VXi AMT', 'Skoda Fabia Classic 1.2 MPI',
       'Maruti Suzuki Stingray VXi', 'Hyundai Elite i20 Magna 1.2',
       'Mahindra Scorpio SLE BS IV', 'Audi A8', 'Audi Q7',
       'Mahindra Scorpio S10', 'Maruti Suzuki Alto 800',
       'Hyundai i20 Sportz 1.2', 'Maruti Suzuki Alto 800 Lx',
       'Maruti Suzuki Vitara Brezza ZDi', 'Maruti Suzuki Alto LX',
       'Mahindra Bolero DI', 'Maruti Suzuki Swift Dzire ZDi',
       'Mahindra Scorpio S10 4WD', 'Maruti Suzuki Swift Vdi BSIII',
       'Maruti Suzuki Wagon R VXi BS III',
       'Maruti Suzuki Wagon R VXi Minor',
       'Toyota Innova 2.0 G 8 STR BS IV', 'Renault Lodgy 85 PS RXL',
       'Skoda Yeti Ambition 2.0 TDI CR 4x2',
       'Maruti Suzuki Baleno Delta 1.2',
       'Renault Duster 110 PS RxZ Diesel Plus',
       'Renault Duster 85 PS RxE Diesel', 'Honda City 1.5 S MT',
       'Maruti Suzuki Dzire', 'Honda Amaze', 'Honda Amaze 1.5 SX i DTEC',
       'Honda City', 'Datsun Redi GO S', 'Maruti Suzuki SX4 ZXI MT',
       'Mitsubishi Pajero Sport Limited Edition',
       'Maruti Suzuki Swift VXi 1.2 ABS BS IV', 'Honda City ZX CVT',
       'Maruti Suzuki Wagon R LX BS IV', 'Tata Indigo eCS LS CR4 BS IV',
       'Volkswagen Polo Highline Exquisite P',
       'I want to sell my car Tata Zest', 'Chevrolet Spark LS 1.0',
       'Renault Duster 110PS Diesel RxZ', 'Mini Cooper S 1.6',
       'Skoda Fabia 1.2L Diesel Ambiente', 'Renault Duster',
       'Mahindra Scorpio S4', 'Mahindra Scorpio VLX 2WD BS IV',
       'Mahindra Quanto C8', 'Ford EcoSport', 'Honda Brio',
       'Volkswagen Vento Highline Plus 1.5 Diesel AT',
       'Hyundai i20 Magna', 'Toyota Corolla Altis Diesel D4DG',
       'Hyundai Verna Transform SX VTVT',
       'Toyota Corolla Altis Petrol Ltd', 'Honda City 1.5 EXi New',
       'Skoda Fabia 1.2L Diesel Elegance', 'BMW 3 Series 320i',
       'Maruti Suzuki A Star Lxi', 'Toyota Etios GD',
       'Ford Figo Diesel EXI Option',
       'Maruti Suzuki Swift Dzire VXi 1.2 BS IV',
       'Chevrolet Beat LT Diesel', 'BMW 7 Series 740Li Sedan',
       'Mahindra XUV500 W8 AWD 2013', 'Hyundai i10 Magna 1.2',
       'Hyundai Verna Fluidic New', 'Maruti Suzuki Swift VXi 1.2 BS IV',
       'Maruti Suzuki Ertiga ZXI Plus', 'Maruti Suzuki Ertiga Vxi',
       'Maruti Suzuki Ertiga VDi', 'Maruti Suzuki Alto LXi BS III',
       'Hyundai Grand i10 Asta 1.1 CRDi', 'Honda Amaze 1.2 S i VTEC',
       'Hyundai i20 Asta 1.4 CRDI 6 Speed', 'Ford Figo Diesel EXI',
       'Maruti Suzuki Eeco 5 STR WITH AC HTR', 'Maruti Suzuki Ertiga ZXi',
       'Maruti Suzuki Esteem LXi BS III', 'Maruti Suzuki Ritz VXI',
       'Maruti Suzuki Ritz LDi', 'Maruti Suzuki Dzire VDI',
       'Toyota Etios Liva G', 'Hyundai i20 Sportz 1.4 CRDI',
       'Chevrolet Spark', 'Nissan Micra XV', 'Maruti Suzuki Swift',
       'Honda Amaze 1.5 S i DTEC', 'Chevrolet Beat', 'Toyota Corolla',
       'Honda City 1.5 V MT', 'Ford EcoSport Trend 1.5L TDCi',
       'Hyundai i20 Asta 1.2', 'Tata Indica V2 eLS',
       'Maruti Suzuki Alto 800 Lxi', 'Hindustan Motors Ambassador',
       'Toyota Corolla Altis 1.8 GL', 'Toyota Corolla Altis 1.8 J',
       'Toyota Innova 2.5 GX BS IV 7 STR',
       'Volkswagen Jetta Highline TDI AT',
       'Volkswagen Polo Comfortline 1.2L P', 'Volkswagen Polo',
       'Mahindra Scorpio', 'Nissan Sunny', 'Hyundai Elite i20',
       'Renault Kwid', 'Mahindra Scorpio VLX Airbag',
       'Chevrolet Spark LT 1.0', 'Datsun Redi GO T O',
       'Maruti Suzuki Swift RS VDI', 'Fiat Punto Emotion 1.2',
       'Hyundai i10 Sportz 1.2', 'Chevrolet Beat LT Opt Diesel',
       'Chevrolet Beat LS Diesel', 'Tata Indigo CS',
       'Maruti Suzuki Swift VDi', 'Hyundai Eon Era Plus',
       'Mahindra XUV500', 'Ford Fiesta', 'Maruti Suzuki Wagon R',
       'Hyundai i20', 'Tata Indigo eCS LX TDI BS III',
       'Hyundai Fluidic Verna 1.6 CRDi SX',
       'Commercial , DZire LDI, 2016, for sale', 'Fiat Petra ELX 1.2 PS',
       'Hyundai Santro Xing XS', 'Maruti Suzuki Ciaz VXi Plus',
       'Maruti Suzuki Zen VX', 'Hyundai Creta 1.6 SX Plus Petrol',
       'Tata indigo ecs LX, 201', 'Mahindra Scorpio SLX',
       'Toyota Innova 2.5 G BS III 8 STR',
       'Maruti Suzuki Wagon R LXI BS IV', 'Tata Nano Cx BSIV',
       'Maruti Suzuki Alto Std BS IV', 'Maruti Suzuki Wagon R LXi BS III',
       'Maruti Suzuki Swift VXI BSIII',
       'Tata Sumo Victa EX 10 by 7 Str BSIII', 'MARUTI SUZUKI DESI',
       'Volkswagen Passat Diesel Comfortline AT',
       'Renault Scala RxL Diesel Travelogue',
       'Hyundai Grand i10 Sportz O 1.2 Kappa VTVT',
       'Hyundai i20 Active 1.2 SX', 'Mahindra Xylo E4',
       'Mahindra Jeep MM 550 XDB', 'Mahindra Bolero SLE BS IV',
       'Force Motors Force One LX ABS 7 STR', 'Maruti Suzuki SX4',
       'Toyota Etios', 'Honda City ZX VTEC',
       'Maruti Suzuki Wagon R LX BS III', 'Honda City VX O MT Diesel',
       'Mahindra Thar CRDe 4x4 AC',
       'Audi A4 1.8 TFSI Multitronic Premium Plus',
       'Mercedes Benz GLA Class 200 CDI Sport',
       'Land Rover Freelander 2 SE', 'Renault Kwid RXT',
       'Tata Aria Pleasure 4X2', 'Mercedes Benz B Class B180 Sports',
       'Datsun GO T O', 'Honda Jazz VX MT',
       'Hyundai i20 Active 1.4L SX O', 'Mini Cooper S',
       'Maruti Suzuki Ciaz ZXI Plus', 'Chevrolet Tavera Neo',
       'Hyundai Eon Sportz', 'Tata Sumo Gold Select Variant',
       'Maruti Suzuki Wagon R 1.0', 'Maruti Suzuki Esteem VXi BS III',
       'Chevrolet Enjoy 1.4 LS 8 STR', 'Maruti Suzuki Wagon R 1.0 VXi',
       'Nissan Terrano XL D Plus', 'Renault Duster 85 PS RxL Diesel',
       'Maruti Suzuki Dzire ZXI', 'Renault Kwid RXT Opt',
       'Maruti Suzuki Maruti 800 Std', 'Renault Kwid 1.0 RXT AMT',
       'Renault Scala RxL Diesel',
       'Hyundai Grand i10 Asta 1.2 Kappa VTVT O',
       'Chevrolet Beat LS Petrol', 'Hyundai Accent GLX', 'Yama',
       'Maruti Suzuki Swift LDi', 'Mahindra TUV300 T4 Plus',
       'Tata Indica V2 Xeta e GLE', 'Tata Indigo CS LS DiCOR',
       'Mahindra Scorpio VLX Special Edition BS III',
       'Tata Indica eV2 LS', 'Honda Accord',
       'Ford EcoSport Titanium 1.5 TDCi', 'Maruti Suzuki Ertiga',
       'Mahindra Scorpio 2.6 CRDe', 'Honda Mobilio',
       'Toyota Corolla Altis', 'Skoda Laura', 'Hyundai Verna Fluidic',
       'Maruti Suzuki Vitara Brezza', 'Tata Manza Aura Quadrajet',
       'Chevrolet Sail UVA Petrol LT ABS',
       'Hyundai Verna Fluidic 1.6 VTVT SX',
       'Audi A4 2.0 TDI 177bhp Premium', 'Hyundai Elantra SX',
       'Mahindra Scorpio VLX 4WD Airbag', 'Mahindra KUV100 K8 D 6 STR',
       'Hyundai Grand i10', 'Hyundai i10', 'Hyundai i20 Active',
       'Datsun Redi GO', 'Toyota Etios Liva', 'Hyundai Accent',
       'Hyundai Verna', 'Toyota Fortuner', 'Hyundai i10 Sportz',
       'Mahindra Bolero Power Plus SLE', 'selling car Ta',
       'Honda City 1.5 V MT Exclusive', 'Chevrolet Spark LT 1.0 Airbag',
       'Tata Indigo eCS VX CR4 BS IV', 'Tata Zest 90',
       'Skoda Rapid Elegance 1.6 TDI CR MT', 'Tata Vista Quadrajet VX',
       'Maruti Suzuki Alto K10 VXi AT', 'Maruti Suzuki Zen LXi BS III',
       'Maruti Suzuki Swift Dzire Tour LDi', 'Honda City ZX EXi',
       'Chevrolet Beat Diesel', 'Maruti Suzuki Swift Dzire car',
       'Hyundai Verna 1.4 VTVT', 'Toyota Innova 2.5 E MS 7 STR BS IV',
       'Maruti Suzuki Maruti 800 Std – Befo',
       'Hyundai Elite i20 Asta 1.4 CRDI',
       'Maruti Suzuki Swift Dzire Tour (Gat',
       'Maruti Suzuki Versa DX2 8 SEATER BSIII',
       'Tata Indigo LX TDI BS III',
       'Volkswagen Vento Konekt Diesel Highline',
       'Mercedes Benz C Class 200 CDI Classic', 'URJE',
       'Hyundai Santro Xing GLS', 'Maruti Suzuki Omni Limited Edition',
       'Hyundai Sonata Transform 2.4 GDi MT',
       'Hyundai Elite i20 Sportz 1.2', 'Honda Jazz S MT',
       'Hyundai Grand i10 Sportz 1.2 Kappa VTVT',
       'Maruti Suzuki Zen LXi BSII',
       'Mahindra Scorpio W Turbo 2.6DX 9 Seater',
       'Swift Dzire Tour 27 Dec 2016 Regis', 'Maruti Suzuki Alto K10 VXi',
       'Hyundai Grand i10 Asta 1.2 Kappa VTVT', 'Mahindra XUV500 W8',
       'Hyundai i20 Magna O 1.2', 'Renault Duster 85 PS RxL Explore LE',
       'Honda Brio V MT', 'Mahindra TUV300 T8',
       'Nissan X Trail Select Variant', 'Ford Ikon 1.3 CLXi NXt Finesse',
       'Toyota Fortuner 3.0 4x4 MT', 'Tata Manza ELAN Quadrajet',
       'Tata zest x', 'Mahindra xyl',
       'Mercedes Benz A Class A 180 Sport Petrol', 'Tata Indigo LS',
       'Hyundai i20 Magna 1.2', 'Used Commercial Maruti Omn',
       'Honda Amaze 1.5 E i DTEC', 'Hyundai Verna 1.6 EX VTVT',
       'BMW 5 Series 520d Sedan', 'Skoda Superb 1.8 TFSI AT',
       'Audi Q3 2.0 TDI quattro Premium', 'Mahindra Bolero DI BSII',
       'Maruti Suzuki Zen Estilo LXI Green CNG',
       'Ford Figo Duratorq Diesel Titanium 1.4',
       'Maruti Suzuki Wagon R VXI BS IV', 'Mahindra Logan Diesel 1.5 DLS',
       'Tata Nano GenX XMA', 'Honda City SV', 'Ford Figo Petrol LXI',
       'Hyundai i10 Magna 1.2 Kappa2', 'Toyota Corolla H2',
       'Maruti Suzuki Swift Dzire Tour VXi', 'Tata Indigo CS eLS BS IV',
       'Hyundai Xcent Base 1.1 CRDi', 'Hyundai Accent Executive Edition',
       'Tata Zest XE 75 PS Diesel', 'Maruti Suzuki Dzire LDI',
       'Tata Sumo Gold LX BS IV', 'Toyota Corolla Altis GL Petrol',
       'Maruti Suzuki Eeco 7 STR', 'Toyota Fortuner 3.0 4x2 MT',
       'Mahindra XUV500 W6', 'Tata Tigor Revotron XZ',
       'Maruti Suzuki 800', 'Honda Mobilio S i DTEC',
       'Hyundai Verna 1.6 CRDI E', 'Maruti Suzuki Omni Select Variant',
       'Tata Indica', 'Hyundai Santro Xing', 'Maruti Suzuki Zen Estilo',
       'Honda Brio VX AT', 'Maruti Suzuki Wagon R Select Variant',
       'Tata Nano Lx BSIV', 'Jaguar XE XE Portfolio',
       'Hyundai Xcent S 1.2', 'Hyundai Eon Magna Plus',
       'Maruti Suzuki Ritz GENUS VXI',
       'Hyundai Grand i10 Magna AT 1.2 Kappa VTVT',
       'Hyundai Eon D Lite Plus', 'Honda Amaze 1.2 VX i VTEC',
       'Maruti Suzuki Estilo VXi ABS BS IV',
       'Maruti Suzuki Vitara Brezza LDi O', 'Toyota Innova 2.0 V',
       'Hyundai Creta 1.6 SX Plus Petrol AT', 'Mahindra Scorpio Vlx BSIV',
       'Mitsubishi Lancer 1.8 LXi', 'Maruti Suzuki Maruti 800 AC',
       'Maruti Suzuki Alto 800 LXI CNG O', 'Ford Fiesta SXi 1.6 ABS',
       'Maruti Suzuki Ritz VDi', 'Maruti Suzuki Estilo LX BS IV',
       'Audi A6 2.0 TDI Premium', 'Maruti Suzuki Alto',
       'Maruti Suzuki Baleno Sigma 1.2', 'Hyundai Verna 1.6 SX VTVT AT',
       'Maruti Suzuki Swift GLAM', 'Hyundai Getz Prime 1.3 GVS',
       'Hyundai Santro', 'Hyundai Getz Prime 1.3 GLX',
       'Chevrolet Beat PS Diesel', 'Ford EcoSport Trend 1.5 Ti VCT',
       'Tata Indica V2 DLG', 'BMW X1 xDrive20d xLine',
       'Honda City 1.5 V AT', 'Tata Nano', 'Chevrolet Cruze LTZ AT',
       'Hyun', 'Maruti Suzuki Swift Dzire VDi', 'Mahindra XUV500 W10',
       'Maruti Suzuki Alto K10 LXi CNG', 'Hyundai Accent GLE',
       'Force Motors One SUV', 'Datsun Go Plus T O',
       'Chevrolet Spark 1.0 LT', 'Toyota Etios Liva GD',
       'Renault Duster 85PS Diesel RxL Optional with Nav',
       'Chevrolet Enjoy', 'BMW 5 Series 530i', 'Chevrolet Cruze LTZ',
       'Jeep Wrangler Unlimited 4x4 Diesel',
       'Hyundai Verna VGT CRDi SX ABS', 'Maruti Suzuki Omni',
       'Maruti Suzuki Celerio VDi', 'Tata Zest Quadrajet 1.3',
       'Tata Indigo CS eLX BS IV', 'Hyundai i10 Era',
       'Tata Indigo eCS LX CR4 BS IV', 'Tata Indigo Marina LS',
       'Commercial Chevrolet Sail Hatchback ca', 'Hyundai Xcent SX 1.2',
       'Tata Nano LX Special Edition', 'Commercial Car Ta',
       'Renault Duster 110 PS RxZ Diesel',
       'Maruti Suzuki Wagon R AX BSIV', 'Maruti Suzuki Alto K10 New',
       'tata Indica', 'Mahindra Xylo E8', 'Tata Manza Aqua Quadrajet',
       'Used bt new conditions ta', 'Renault Kwid 1.0', 'Sale tata',
       'Tata Venture EX 8 STR', 'Maruti Suzuki Swift Dzire Tour LXi',
       'Maruti Suzuki Alto LX BSII', 'Skoda Octavia Classic 1.9 TDI MT',
       'Maruti Suzuki Omni LPG BS IV', 'Tata Sumo Gold EX BS IV',
       'Tata indigo 2017 top model..', 'Hyundai Verna 1.6 CRDI SX',
       'Mahindra Scorpio SLX 2.6 Turbo 8 Str', 'Ford Ikon 1.6 Nxt',
       'Tata indigo', 'Toyota Innova 2.5 V 7 STR', 'Nissan Sunny XL',
       'Maruti Suzuki Swift VDi BS IV',
       'very good condition tata bolts are av', 'Toyota Innova 2.0 G4',
       'Sale Hyundai xcent commerc', 'Maruti Suzuki Swift VDi ABS',
       'Hyundai Elite i20 Asta 1.2', 'Volkswagen Polo Trendline 1.5L D',
       'Toyota Etios Liva Diesel', 'Maruti Suzuki Ciaz ZXi Plus RS',
       'Hyundai Elantra 1.8 S', 'Ford EcoSport Trend 1.5L Ti VCT',
       'Jaguar XF 2.2 Diesel Luxury',
       'Audi Q5 2.0 TDI quattro Premium Plus', 'BMW 3 Series 320d Sedan',
       'Maruti Suzuki Swift ZXi 1.2 BS IV', 'BMW X1 sDrive20d',
       'Maruti Suzuki S Cross Sigma 1.3', 'Maruti Suzuki Ertiga LDi',
       'Volkswagen Vento Comfortline Petrol', 'Mahindra KUV100',
       'Maruti Suzuki Swift Dzire Tour VDi', 'Mahindra Scorpio 2.6 SLX',
       'Maruti Suzuki Omni 8 STR BS III',
       'Volkswagen Jetta Comfortline 1.9 TDI AT', 'Volvo S80 Summum D4',
       'Toyota Corolla Altis VL AT Petrol',
       'Mitsubishi Pajero Sport 2.5 AT', 'Chevrolet Beat LT Petrol',
       'BMW X1', 'Mercedes Benz C Class C 220 CDI Avantgarde',
       'Volkswagen Vento Comfortline Diesel', 'Tata Indigo CS GLS',
       'Ford Figo Petrol Titanium', 'Honda City ZX GXi',
       'Maruti Suzuki Wagon R Duo Lxi', 'Maruti Suzuki Zen LX BSII',
       'Renault Duster RxL Petrol', 'Maruti Suzuki Baleno Zeta 1.2',
       'Honda WR V S MT Petrol', 'Renault Duster 110 PS RxL Diesel',
       'Mahindra Scorpio LX BS III',
       'Maruti Suzuki SX4 Celebration Diesel',
       'Audi A3 Cabriolet 40 TFSI',
       'I want to sell my commercial car due t',
       'Hyundai Santro AE GLS Audio',
       'i want sale my car.no emi....uber atta', 'Tata ZEST 6 month old',
       'Mahindra Xylo D2 BS IV', 'Hyundai Getz GLE',
       'Hyundai Creta 1.6 SX', 'Hyundai Santro Xing XL AT eRLX Euro III',
       'Hyundai Santro Xing XL eRLX Euro III',
       'Tata Indica V2 DLS BS III', 'Honda City 1.5 E MT',
       'Nissan Micra XL', 'Honda City 1.5 S Inspire',
       'Tata Indica eV2 eXeta eGLX', 'Maruti Suzuki Omni E 8 STR BS IV',
       'MARUTI SUZUKI ERTIGA F', 'Hyundai Verna 1.6 CRDI SX Plus AT',
       'Chevrolet Tavera LS B3 10 Seats BSII', 'Tata Tiago Revotron XM',
       'Tata Tiago Revotorq XZ', 'Tata Nexon', 'Tata',
       'Hindustan Motors Ambassador Classic Mark 4 – Befo',
       'Ford Fusion 1.4 TDCi Diesel',
       'Fiat Linea Emotion 1.4 L T Jet Petrol',
       'Ford Ikon 1.3 Flair Josh 100', 'Tata Indica V2 LS',
       'Mahindra Xylo D2', 'Hyundai Eon Magna',
       'Tata Sumo Grande MKII GX', 'Volkswagen Polo Highline1.2L P',
       'Tata Tiago Revotron XZ', 'Tata Indigo eCS',
       '2012 Tata Sumo Gold f', 'Mahindra Xylo E8 BS IV',
       'Well mentained Tata Sumo',
       'all paper updated tata indica v2 and u',
       'Maruti Ertiga showroom condition with',
       '7 SEATER MAHINDRA BOLERO IN VERY GOOD', '9 SEATER MAHINDRA BOL',
       'scratch less Tata I', 'Maruti Suzuki swift dzire for sale in',
       'Commercial Chevrolet beat for sale in',
       'urgent sell my Mahindra qu', 'Tata Sumo Gold FX BSIII',
       'sell my car Maruti Suzuki Swif',
       'Maruti Suzuki Swift Dzire good car fo', 'Hyunda',
       'Commercial Maruti Suzuki Alto Lxi 800', 'urgent sale Ta',
       'Maruti Suzuki Alto vxi t', 'tata', 'TATA INDI', 'Hyundai Creta',
       'Tata Bolt XM Petrol', 'Hyundai Venue', 'Maruti Suzuki Ritz',
       'Renault Lodgy', 'Hyundai i20 Asta',
       'Maruti Suzuki Swift Select Variant', 'Tata Indica V2 DLX BS III',
       'Mahindra Scorpio VLX 2.2 mHawk Airbag BSIV',
       'Toyota Innova 2.5 E 8 STR', 'Mahindra KUV100 K8 6 STR',
       'Datsun Go Plus', 'Ford Endeavor 4x4 Thunder Plus',
       'Tata Indica V2', 'Hyundai Santro Xing GL',
       'Toyota Innova 2.5 Z Diesel 7 Seater',
       'Any type car avaiabel hare...comercica', 'Maruti Suzuki Alto AX',
       'Mahindra Logan', 'Maruti Suzuki 800 Std BS III',
       'Chevrolet Sail 1.2 LS',
       'Volkswagen Vento Highline Plus 1.5 Diesel', 'Tata Manza',
       'Toyota Innova 2.0 G1 Petrol 8seater', 'Toyota Etios G',
       'Toyota Qualis', 'Mahindra Quanto C4', 'Maruti Suzuki Swift Dzire',
       'Hyundai i20 Select Variant', 'Honda City VX Petrol',
       'Hyundai Getz', 'Mercedes Benz C Class 200 K MT', 'Skoda Fabia',
       'Maruti Suzuki Alto 800 Select Variant',
       'Maruti Suzuki Ritz VXI ABS', 'tata zest 2017 f',
       'Tata Indica V2 DLE BS III', 'Ta', 'Tata Zest XM Diesel',
       'Honda Amaze 1.2 E i VTEC', 'Chevrolet Sail 1.2 LT ABS']

name=st.selectbox('name',sorted(naam))
company=st.selectbox('company',sorted(comp))
year=st.selectbox('year',sorted(yr))
kms_driven=int(st.number_input('kms_driven'))
fuel_type=st.selectbox('fuel_type',sorted(fuel))

input_df = pd.DataFrame(
    {'name': [name], 'company': [company], 'year':[year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
n=pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([name,company,year,kms_driven,fuel_type]).reshape(1,5)))
#result = pipe.predict(input_df)

if st.button('Predict'):
    st.header('Price of Car is '+str(int(n)))