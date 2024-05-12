import warnings

warnings.filterwarnings('ignore')
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from model import KPMER
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg, adj_entity, adj_relation = data[7], data[8], data[9]
    model = KPMER(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_auc_sum = 0
        train_acc_sum = 0
        eval_auc_sum = 0
        eval_acc_sum = 0
        test_auc_sum = 0
        test_acc_sum = 0

        best_avg_f1 = 0.0
        best_topk_results = []

        for step in range(args.n_epochs):
            # RS training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                # end = min(start + args.batch_size, train_data.shape[0])
                # batch_data = train_data[start:end]
                # if len(batch_data) < args.batch_size:
                #     break  # 如果数据不足一个 batch size，则跳出循环
                _, loss = model.train_rs(sess, get_feed_dict_for_rs(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(loss)

            # KGE training
            if step % args.kge_interval == 0:
                np.random.shuffle(kg)
                start = 0
                # while start + args.batch_size <= kg.shape[0]:
                #     _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, train_data, start,
                #                                                           start + args.batch_size))
                #     start += args.batch_size
                while start < kg.shape[0]:
                    _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, start, start + args.batch_size))
                    start += args.batch_size
                    # if show_loss:
                    #     print(rmse)

            # CTR evaluation
            train_auc, train_acc = model.eval(sess,  get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(sess,  get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            test_auc, test_acc = model.eval(sess,  get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            # train_auc, train_acc = ctr_eval(sess, model, train_data, args.batch_size)
            # eval_auc, eval_acc = ctr_eval(sess, model, eval_data, args.batch_size)
            # test_auc, test_acc = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
            train_auc_sum += train_auc
            train_acc_sum += train_acc
            eval_auc_sum += eval_auc
            eval_acc_sum += eval_acc
            test_auc_sum += test_auc
            test_acc_sum += test_acc
            # top-K evaluation
            if show_topk:
                precision, recall, f1 = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                # 计算并输出当前 epoch 的平均值
                avg_f1 = np.mean(f1)
                print('Epoch %d: f1=%.4f' % (args.n_epochs, avg_f1))

                # 保存最好的 epoch
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_topk_results = list(zip(k_list, precision, recall, f1))

        for k, precision, recall, f1 in best_topk_results:
            print('Top-%d: precision=%.4f, recall=%.4f, f1=%.4f' % (k, precision, recall, f1))
        #         print("TopK")
        #         print()
        #         print('precision: ', end='')
        #         for i in precision:
        #             print('%.4f\t' % i, end='')
        #         print()
        #         print('recall: ', end='')
        #         for i in recall:
        #             print('%.4f\t' % i, end='')
        #         print()
        #         print('f1: ', end='')
        #         for i in f1:
        #             print('%.4f\t' % i, end='')
        #         print('\n')
        # print(' train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
        #       % (train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
        train_avg_auc = train_auc_sum / args.n_epochs
        train_avg_acc = train_acc_sum / args.n_epochs
        eval_avg_auc = eval_auc_sum / args.n_epochs
        eval_avg_acc = eval_acc_sum / args.n_epochs
        test_avg_auc = test_auc_sum / args.n_epochs
        test_avg_acc = test_acc_sum / args.n_epochs
        print(
            'train_avg_auc: %.4f  train_avg_acc: %.4f    eval_avg_auc: %.4f  eval_avg_acc: %.4f    test_avg_auc: %.4f  test_avg_acc: %.4f'
            % (train_avg_auc, train_avg_acc, eval_avg_auc, eval_avg_acc, test_avg_auc, test_avg_acc))

def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.head_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict

# def get_feed_dict_for_kge(model, kg, data, start, end):
#     feed_dict = {model.user_indices: data[start:end, 0],
#                  model.item_indices: kg[start:end, 0],
#                  model.head_indices: kg[start:end, 0],
#                  model.tail_indices: kg[start:end, 2],
#                  model.relation_indices: kg[start:end, 1]}
#     return feed_dict

def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {model.item_indices: kg[start:end, 0],
                 model.head_indices: kg[start:end, 0],
                 model.relation_indices: kg[start:end, 1],
                 model.tail_indices: kg[start:end, 2]}
    return feed_dict


# def ctr_eval(sess, model, data, batch_size):
#     start = 0
#     auc_list = []
#     acc_list = []
#     while start + batch_size <= data.shape[0]:
#         auc, acc = model.eval(sess, get_feed_dict_for_rs(model, data, start, start + batch_size))
#         auc_list.append(auc)
#         acc_list.append(acc)
#         start += batch_size
#     return float(np.mean(auc_list)), float(np.mean(acc_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

    return precision, recall, f1

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
