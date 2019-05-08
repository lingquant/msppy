mkdir result
python Markovian_MC.py SA50 10 3
python Markovian_MC.py SA100 10 3
python Markovian_MC.py RSA50 10 3
python Markovian_MC.py RSA100 10 3
python Markovian_MC.py SAA50 100 3000
python TS.py 50 29 3
python TS.py 100 40 3
python compare.py
