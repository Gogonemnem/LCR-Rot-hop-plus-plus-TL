from main_transfer import make_model, pre_proc
import tensorflow as tf

asp_accuracy = []
doc_accuracy = []
asp_f1 = []
doc_f1 = []
lambdas = []

for i in range(0, 10):
    print("Building PRET+MULT model...")
    l = 0.1 + i*0.05
    hyper_param_mult = {'regularizer': tf.keras.regularizers.L1L2(l1=1e-07, l2=0.001),
                        'lr': 0.001,
                        'drop_1': 0.3,
                        'drop_2': 0.6,
                        'hidden_units': 250,
                        'lambda': l}
    model = make_model(settings=[True, True, False],
                                      pret_train_path=,
                                      pret_test_path=,
                                      mult_asp_train_path=,
                                      mult_asp_test_path=,
                                      mult_doc_train_path=,
                                      mult_doc_test_path=,
                                      asp_train_path=,
                                      asp_test_path=,
                                      h=hyper_param_mult,
                                      pret_model_path=)
    x_val, y_val = pre_proc(asp_path=, doc_path=)
    result = model.evaluate(x_val, y_val)

    lambdas.append(l)
    asp_accuracy.append(result[3])
    asp_f1.append(result[4])
    doc_accuracy.append(result[5])
    doc_f1.append(result[6])
    del model
    tf.keras.backend.clear_session()

print('lambdas: ', lambdas)
print('asp accuracy: ', asp_accuracy)
print('asp f1: ', asp_f1)
print('doc accuracy: ', doc_accuracy)
print('doc f1: ', doc_f1)
