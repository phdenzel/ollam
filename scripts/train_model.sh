epochs=50
learning_rate=0.001
batch_size=64

pipenv run ollam --train --epochs ${epochs} --lr ${learning_rate} --bs ${batch_size} 2>&1 | tee ~/.ollam/tr_Adam_${epochs}_${learning_rate}_${batch_size}.log
