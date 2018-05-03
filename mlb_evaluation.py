import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

class MLB_Evaluation(object):

    def __init__(self, data_truth_lbls=None, result_list=None, y_truth_list=None, y_pred_list=None):
        assert (data_truth_lbls and result_list) or (y_truth_list and y_pred_list), "invalid input!"
        if data_truth_lbls and result_list:
            self.id_n_lbls = data_truth_lbls
            self.raw_data = {}
            self.y_p = []
            self.binarizer = MultiLabelBinarizer()
            self.y_bin = self.binarizer.fit_transform(data_truth_lbls)
            self.lbl_seq = self.binarizer.classes_
            # print len(self.lbl_seq) #
            self.lbl_n_posid = {} # derived from self.lbl_seq
            for i,j in enumerate(self.lbl_seq):
                self.lbl_n_posid[j] = i
            self.num_lbl = len(self.lbl_seq)

            # add raw_data
            self.raw_data = result_list
            normalize = False
            for res in result_list:
                temp = [0 for _ in range(self.num_lbl)]
                for lbl, proba in res:
                    # print lbl in self.lbl_n_posid # labels in result list may not exist in truth because result list contains every possible labels
                    if lbl not in self.lbl_n_posid:
                        continue
                    temp[self.lbl_n_posid[lbl]] = proba
                    if proba < 0:
                        normalize = True
                self.y_p.append(temp)

            # normalize if needed
            if normalize:
                y_p_normalized = []
                for i in self.y_p:
                    temp = []
                    for j in i:
                        temp.append((j + 1) / 2.0)
                    y_p_normalized.append(temp)
                self.y_p = np.array(y_p_normalized)
            else:
                self.y_p = np.array(self.y_p)

        elif y_truth_list and y_pred_list:
            self.num_lbl = len(y_truth_list[0])

            self.y_bin = np.array(y_truth_list)
            self.y_p = np.array(y_pred_list)

    def Avg_Ndcg(self):
        ans = 0
        num_record_used = 0
        for i in range(len(self.y_p)):
            score = self.__custom_ndcg_score(self.y_bin[i], self.y_p[i])
            if score >= 0: # only consider non-zero
                ans += score
                num_record_used += 1
        return float(ans) / num_record_used

    def Coverage_Error(self):
        return coverage_error(self.y_bin, self.y_p)

    def Label_Ranking_Average_Precision_Score(self):
        return label_ranking_average_precision_score(self.y_bin, self.y_p)

    def Label_Ranking_Loss(self):
        return label_ranking_loss(self.y_bin, self.y_p)

    def Recall_at_k(self):
        recall_arr = [0 for i in range(self.num_lbl)]
        for idx in range(len(self.y_bin)):
            total = float(sum(self.y_bin[idx]))
            zipped = zip(self.y_p[idx], self.y_bin[idx])
            zipped.sort(key=lambda x: x[0], reverse=True)
            counter = 0
            # print zipped
            for i, tup in enumerate(zipped):
                if tup[1]:
                    counter += 1
                if counter < total:
                    recall_arr[i] += counter / total
                    # print counter / total
                else:
                    recall_arr[i] += 1.0
                    # print 1.0
        recall_arr = [i / len(self.y_p) for i in recall_arr]
        return recall_arr

    def Accuracy_Score(self, threshold): # interesting, high baseline

        processed_y_p = self.__threshold_process(threshold)

        ans = 0.0
        for i in range(len(processed_y_p)):
            if list(processed_y_p[i]) == list(self.y_bin[i]):
                ans += 1
        return ans / len(processed_y_p)

    def Precision_Score(self, threshold):

        processed_y_p = self.__threshold_process(threshold)
        precision_list = self.__calc_precision_list(processed_y_p)

        return sum(precision_list) / len(precision_list)

    def Recall_Score(self, threshold):

        processed_y_p = self.__threshold_process(threshold)
        recall_list = self.__calc_recall_list(processed_y_p)

        return sum(recall_list) / len(recall_list)

    def F1_Score(self, threshold):

        processed_y_p = self.__threshold_process(threshold)
        precision_list = self.__calc_precision_list(processed_y_p)
        recall_list = self.__calc_recall_list(processed_y_p)

        f1_list = []
        for i in range(len(processed_y_p)):
            if precision_list[i] + recall_list[i]:
                f1_list.append(2 * precision_list[i] * recall_list[i] / (precision_list[i] + recall_list[i]))
            else:
                f1_list.append(0)

        return sum(f1_list) / len(f1_list)

    def __calc_recall_list(self, processed_y_p):
        ans = []
        for i in range(len(processed_y_p)):
            tp = 0.0
            tp_fn = 0.0
            for j in range(len(processed_y_p[i])):
                if processed_y_p[i][j] == 1 and self.y_bin[i][j] == 1:
                    tp += 1
                    tp_fn += 1
                elif self.y_bin[i][j] == 1:
                    tp_fn += 1
            ans.append(tp / tp_fn)
        return ans

    def __calc_precision_list(self, processed_y_p):
        ans = []
        for i in range(len(processed_y_p)):
            tp = 0.0
            tp_fp = 0.0
            for j in range(len(processed_y_p[i])):
                if processed_y_p[i][j] == 1 and self.y_bin[i][j] == 1:
                    tp += 1
                    tp_fp += 1
                elif processed_y_p[i][j] == 1:
                    tp_fp += 1
            if tp_fp > 0:
                ans.append(tp / tp_fp)
            else:
                ans.append(0)
        return ans

    def __threshold_process(self, threshold):
        processed_y_p = []
        for i in self.y_p:
            temp = []
            for j in i:
                if j >= threshold:
                    temp.append(1)
                else:
                    temp.append(0)
            processed_y_p.append(temp)
        processed_y_p = np.array(processed_y_p)
        return processed_y_p

    def __custom_ndcg_score(self, y, y_pred):
        """
        Calculate ndcg score for two formatted input vectors.
        :param y: Binary vector, 1 for correct position, e.g. [0, 1, 1, 0, 0]
        :param y_pred: Same-dimension vector, proba/score for each position, e.g. [0.1, 0.8, 0.9, 0.3, 0.5]
        :return: NDCG score, a single number
        """
        # get index of all correct answers, and get smallest number among them
        correct_index = set()
        smallest = 1
        for idx, label in enumerate(y):
            if label == 1:
                correct_index.add(idx)
                if y_pred[idx] < smallest:
                    smallest = y_pred[idx]

        # include value for correct indices and valued larger than smallest correct
        index_and_val = {}
        for idx, val in enumerate(y_pred):
            if idx in correct_index:
                index_and_val[idx] = y_pred[idx]
            elif val > smallest:
                index_and_val[idx] = y_pred[idx]

        index_and_val = sorted(index_and_val.items(), key = lambda x: x[1], reverse=True)

        #
        pred_list = []
        for idx, val in index_and_val:
            if idx in correct_index:
                pred_list.append(1)
            else:
                pred_list.append(0)

        # idcg
        idcg = 0
        for idx in range(len(correct_index)):
            idcg += 1 / np.log2(idx + 1 + 1)

        # dcg
        dcg = 0
        for idx, score in enumerate(pred_list):
            dcg += score / np.log2(idx + 1 + 1)

        if idcg == 0: # if no correct index
            return 0

        return dcg / idcg

def evaluation(truth1, pred1, truth2, pred2, save=False, saveid='mlb0'):
    MLB = MLB_Evaluation(truth1, pred1, truth2, pred2)
    avg_ndcg = MLB.Avg_Ndcg()
    cov_err = MLB.Coverage_Error()
    lrap = MLB.Label_Ranking_Average_Precision_Score()
    lrl = MLB.Label_Ranking_Loss()
    recall_arr = MLB.Recall_at_k()
    print "ndcg: " + str(avg_ndcg)
    print "coverage error: " + str(cov_err)
    print "LRAP score: " + str(lrap)
    print "label ranking loss: " + str(lrl)
    print "recall @k: " + str(recall_arr)
    # print "accuracy score @%s: " % str(threshold) + str(MLB.Accuracy_Score(threshold))
    # print "precision score @%s: " % str(threshold) + str(MLB.Precision_Score(threshold))
    # print "recall score @%s: " % str(threshold) + str(MLB.Recall_Score(threshold))
    # print "f1 score @%s: " % str(threshold) + str(MLB.F1_Score(threshold))

    if save:

        with open(saveid + ".txt", "w") as f:
            f.write("ndcg: " + str(avg_ndcg))
            f.write("\n")
            f.write("coverage error: " + str(cov_err))
            f.write("\n")
            f.write("LRAP score: " + str(lrap))
            f.write("\n")
            f.write("label ranking loss: " + str(lrl))
            f.write("\n")
            f.write("recall @k: " + str(recall_arr))
            f.write("\n")


# unit tests
# pred = [[0.1,0.2,0.9,0.8,0.7], [0.8,0.7,0.6,0.5,0.2,0.1], [0.8,0.9,0,0,0], [0,0,0,0,0.8]]
# truth = [[0,0,1,1,1], [0,1,0,1,0], [1,0,0,0,0], [0,0,0,0,1]]
#
# mlb = MLB_Evaluation(None, None, truth, pred)
# mlb.Recall_at_k()


#
# # if __name__ == "__main__":
#
# # dev_test_ids_10_seq = json.loads(open("dev_test_ids_10_seq.json").read())
# dev_test_themes_truth_10_seq = json.loads(open("dev_test_themes_truth_10_seq.json").read())
#
#
#
#
# dev_train_index_90 = json.loads(open("tfidf_series/dev_train_index_90.json").read())
# dev_test_index_10 = json.loads(open("tfidf_series/dev_test_index_10.json").read())
#
# test_themes_truth_10_seq_tfidf = []
# for i in dev_test_index_10:
#     id = id_list[i]
#     test_themes_truth_10_seq_tfidf.append(id_and_themes_dev_underscore[id])
#
# with open("test_themes_truth_10_seq_tfidf.json", "w") as f:
#     f.write(json.dumps(test_themes_truth_10_seq_tfidf))
#
# test_themes_truth_10_seq_fasttext = []
# for i in test_index_ft:
#     id = id_list_all[i]
#     test_themes_truth_10_seq_fasttext.append(id_and_themes_dev_underscore[id])
#
# with open("test_themes_truth_10_seq_fasttext.json", "w") as f:
#     f.write(json.dumps(test_themes_truth_10_seq_fasttext))
#
#
# # tfidf series
# result_lbl_n_prob_best_tfidf_rf_id5x_10 = json.loads(open("results/result_lbl_n_prob_best_tfidf_rf_id5x_10.json").read())
# result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10 = json.loads(open("results/result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10.json").read())
# evaluation(test_themes_truth_10_seq_tfidf, result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10, None, None)
#
#
# result_lbl_n_prob_best_FastText_sup_id1x_10 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_sup_id1x_10.json").read())
# result_lbl_n_prob_best_FastText_RidgeCV_id2x_10 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_RidgeCV_id2x_10.json").read())
# result_lbl_n_prob_best_FastText_rf_id4x_10 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_rf_id4x_10.json").read())
# evaluation(test_themes_truth_10_seq_fasttext, result_lbl_n_prob_best_FastText_rf_id4x_10, None, None)
#
#
# bin_y_dev_ready_test = []
# for idx in test_index:
#     bin_y_dev_ready_test.append(bin_y_dev_ready[idx])
#
#
#
# evaluation(None, None, bin_y_dev_ready_test, best_result_record)
#
#
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_FastText_sup_id1x_10, None, None)
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_FastText_RidgeCV_id2x_10)
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10) #
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_FastText_rf_id4x_10)
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_tfidf_rf_id5x_10)
#
#
#
#
#
# id_and_themes_dev_underscore = json.loads(open("id_and_themes_dev_underscore.json").read())
# dev_test_themes_truth_100_seq = []
# for i in id_list:
#     dev_test_themes_truth_100_seq.append(id_and_themes_dev_underscore[i])
#
# with open("dev_test_themes_truth_100_seq.json", "w") as f:
#     f.write(json.dumps(dev_test_themes_truth_100_seq))
#
#
# result_ndcg_best_tfidf_RidgeCV_id3x_10 = json.loads(open("results/result_ndcg_best_tfidf_RidgeCV_id3x_10.json").read())
# result_ndcg_best_tfidf_RidgeCV_id3x_10_try = json.loads(open("result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10_try.json").read())
#
#
#
# evaluation(dev_test_themes_truth_10_seq, result_ndcg_best_tfidf_RidgeCV_id3x_10_try)
#
#
#
# result_ndcg_best_tfidf_rf_id5x_10_D3_2 = json.loads(open("result_ndcg_best_tfidf_rf_id5x_10_D3_2.json").read())
#
#
# result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10_xxx = json.loads(open("result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10_xxx.json").read())
# evaluation(dev_test_themes_truth_10_seq, result_lbl_n_prob_best_tfidf_RidgeCV_id3x_10_xxx, 0.2)
#
#
#
# detail_proba_rf_tfidf_1000_best_correct_id5_unprocessed = json.loads(open("tfidf_series/detail_proba_rf_tfidf_1000_best_correct_id5_unprocessed.json").read())
# tfidf_truth_y_bin_seq_list = json.loads(open("tfidf_series/tfidf_truth_y_bin_seq_list.json").read())
# evaluation(None, None, tfidf_truth_y_bin_seq_list, detail_proba_rf_tfidf_1000_best_correct_id5_unprocessed)
#
#
# result_lbl_n_prob_best_tfidf_RidgeCV_id3 = json.loads(open("tfidf_series/result_lbl_n_prob_best_tfidf_RidgeCV_id3.json").read())
# evaluation(dev_test_themes_truth_100_seq, result_lbl_n_prob_best_tfidf_RidgeCV_id3, None, None)
#
# y_rawlabels_dev = json.loads(open("y_rawlabels_dev.json").read())
# FastText_truth_y_bin_seq_list = json.loads(open("fasttext_series/FastText_truth_y_bin_seq_list.json").read())
#
# result_lbl_n_prob_best_FastText_sup_id1 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_sup_id1.json").read())
# result_lbl_n_prob_best_FastText_RidgeCV_id2 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_RidgeCV_id2.json").read())
# result_lbl_n_prob_best_FastText_rf_id4 = json.loads(open("fasttext_series/result_lbl_n_prob_best_FastText_rf_id4.json").read())
# evaluation(y_rawlabels_dev, result_lbl_n_prob_best_FastText_rf_id4, None, None)
#
#
#
#
#
#
# train_str_ids_90 = []
# for i in train_index:
#     train_str_ids_90.append(id_list[i])
#
# with open("train_str_ids_90.json", "w") as f:
#     f.write(json.dumps(train_str_ids_90))
#
# test_str_ids_10 = []
# for i in test_index:
#     test_str_ids_10.append(id_list[i])
#
# with open("test_str_ids_10.json", "w") as f:
#     f.write(json.dumps(test_str_ids_10))
#
# train_index_ft = []
# for id in train_str_ids_90:
#     train_index_ft.append(id_list_all.index(id))
#
# with open("train_index_ft.json", "w") as f:
#     f.write(json.dumps(train_index_ft))
#
# test_index_ft = []
# for id in test_str_ids_10:
#     test_index_ft.append(id_list_all.index(id))
#
# with open("test_index_ft.json", "w") as f:
#     f.write(json.dumps(test_index_ft))
#
#





# id_and_themes_dev = json.loads(open("id_and_themes_dev.json").read())



# dev_test_set_10_n_themes = json.loads(open("dev_test_set_10_n_themes.json").read())
#
# dev_test_set_10_list_themes_truth = []
# for i in dev_test_set_ids_10:
#     dev_test_set_10_list_themes_truth.append(dev_test_set_10_n_themes[i])
#
# with open("dev_test_set_10_list_themes_truth.json", "w") as f:
#     f.write(json.dumps(dev_test_set_10_list_themes_truth))

# mlb = MultiLabelBinarizer()
# bin_test = mlb.fit_transform(dev_test_set_10_list_themes_truth)

# dev_test_set_10_list_themes_truth_2 = []
# for i in dev_test_set_10_list_themes_truth:
#     tmp = []
#     for j in i:
#         tmp.append("_".join(j.split()))
#     dev_test_set_10_list_themes_truth_2.append(tmp)
#
# with open("dev_test_set_10_list_themes_truth.json", "w") as f:
#     f.write(json.dumps(dev_test_set_10_list_themes_truth_2))

# ####
# # correct
# ids_raw_dev = json.loads(open("ids_raw_dev.json").read())
# dev_test_index_10 = json.loads(open("dev_test_index_10.json").read())
#
# dev_test_ids_10_new = []
# for i in dev_test_index_10:
#     dev_test_ids_10_new.append(ids_raw_dev[i])
#
# with open("dev_test_ids_10_seq.json", "w") as f:
#     f.write(json.dumps(dev_test_ids_10_new))
#
# id_and_themes_dev = json.loads(open("id_and_themes_dev.json").read())
# dev_test_set_10_list_themes_truth_new = []
# for id in dev_test_ids_10_new:
#     temp = []
#     for i in id_and_themes_dev[id]:
#         temp.append("_".join(i.split()))
#     dev_test_set_10_list_themes_truth_new.append(temp)
#
#
# with open("dev_test_themes_truth_10_seq.json", "w") as f:
#     f.write(json.dumps(dev_test_set_10_list_themes_truth_new))

# return [MLB.Avg_Ndcg(),
#        MLB.Coverage_Error(),
#        MLB.Label_Ranking_Average_Precision_Score(),
#        MLB.Label_Ranking_Loss(),
#        MLB.Accuracy_Score(threshold),
#        MLB.Precision_Score(threshold),
#        MLB.Recall_Score(threshold),
#        MLB.F1_Score(threshold)]