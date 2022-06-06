#测试用的文件，不是正式程序

# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
#
# #get_batches函数会调用get_target函数
# def get_target(words, idx, window_size=5):
#     '''Get a list of words in a window around an index. '''
#     #words：单词列表；idx：input word的索引号
#
#     target_window = np.random.randint(1, window_size + 1)
#     start_point = idx - target_window if (idx - target_window) > 0 else 0
#     end_point = idx + target_window
#     target_words = set(words[start_point:idx] + words[idx + 1:end_point + 1])
#     return list(target_words)
# # 我们定义了一个get_targets函数，接收一个单词索引号，基于这个索引号去查找单词表中对应的上下文（默认window_size=5）。
# # 请注意这里有一个小trick，我在实际选择input word上下文时，使用的窗口大小是一个介于[1, window_size]区间的随机数。
# # 这里的目的是让模型更多地去关注离input word更近词。
#
# def get_batches(words, batch_size, window_size=5):
#     ''' 构建一个获取batch的生成器, Create a generator of word batches as a tuple (inputs, targets) '''
#
#     n_batches = len(words) // batch_size
#     # // 表示整除
#
#     # only full batches仅取full batches
#     words = words[:n_batches * batch_size]
#
#     for idx in range(0, len(words), batch_size):
#         x, y = [], []
#         batch = words[idx:idx + batch_size]
#         for i in range(len(batch)):
#             batch_x = batch[i]
#             batch_y = get_target(batch, i, window_size)
#
#             #由于一个inputword会对应多个outputword
#             y.extend(batch_y)
#             x.extend([batch_x] * len(batch_y))
#         yield x, y
#
#         #yield函数详解见：https://blog.csdn.net/mieleizhi0522/article/details/82142856
#
#
#
# test_sentence2 = """When forty winters shall besiege thy brow,
# And dig deep trenches in thy beauty's field,
# Thy youth's proud livery so gazed on now,
# Will be a totter'd weed of small worth held:
# Then being asked, where all thy beauty lies,
# Where all the treasure of thy lusty days;
# To say, within thine own deep sunken eyes,
# Were an all-eating shame, and thriftless praise.
# How much more praise deserv'd thy beauty's use,
# If thou couldst answer 'This fair child of mine
# Shall sum my count, and make my old excuse,'
# Proving his beauty by succession thine!
# This were to be new made when thou art old,
# And see thy blood warm when thou feel'st it cold.""".split()
#
# test_sentence = """008000 000547 005725 003280 005070 005870 004980 003465 004250 004960 009580 004150 002810 000660 000370 010130 015860 051900 010780 014160 004135 002780 005940 025540 139130 008040 011420 002140 016450 001740 009070 008350 004989 005440 018470 000540 000670 005390 005385 004090 001200 008260 011155 004545 009160 008700 001800 000145 003545 020760 003410 069960 138930 007630 009450 006370 000390 004000 005380 067830 002700 014790 025530 001140 000060 001527 105560 035000 086790 012630 007700 002200 001460 016360 002840 010120 015360 009830 006260 004970 017040 003550 023960 001120 008060 001450 033240 036580 005950 006060 030210 008970 014280 009240 007815 005620 000545 002720 003540 003925 009310 002550 000120 016880 002760 006220 002460 006840 024720 004105 000640 001795 033180 003470 002790 011160 068870 002350 014825 007310 015230 008770 066570 003690 000150 004490 009180 003070 005305 006200 008930 002900 003960 009410 037710 016590 007160 008730 005690 002795 058430 069260 017960 006120 001745 006360 036530 055550 017900 000240 008110 002310 016380 002240 011760 003720 000890 004020 015260 009835 006340 008560 001510 011000 016385 005450 012600 001260 001070 005820 006650 002360 002005 005389 005430 006740 003060 000700 033920 005945 006345 014130 013520 175330 018880 011810 002710 031440 016090 002420 000810 001525 010820 004710 002270 005387 006570 034020 004370 032640 007460 006400 003075 016710 001720 005090 030790 004060 000880 008600 023800 001520 007860 006660 023450 011700 000320 001500 033780 006405 003010 003580 011300 000180 033250 000720 009970 003419 003000 004985 014990 000270 005030 008250 011690 001420 007575 026890 014680 011090 002920 047040 001080 010770 033530 030000 007610 066575 013000 011150 002355 012200 015020 014285 071050 004170 007590 005880 010060 002600 004870 000050 003830 004835 003555 009440 004910 005930 009150 006800 004920 000970 000990 001515 019490 013700 000910 008420 015350 004690 002170 012610 009680 003240 000230 004800 035150 000490 008870 004560 005800 003475 001067 001880 001230 030720 004720 011930 000020 001060 001275 005257 003230 002030 005980 000885 030200 000500 030610 007980 010620 003850 001790 004365 013570 001465 009270 011280 003415 004100 000325 011785 006490 000860 014440 003680 017370 000760 002960 021960 009420 000157 002630 017550 003520 016580 025560 004080 019175 010600 002620 000220 004415 005810 012690 007340 005490 001290 010040 019180 003300 005680 015760 005255 001570 024900 001620 001750 009470 005740 001045 000430 000075 007190 051915 051905 000100 008355 068875 003495 000215 005960 003530 003480 003160 004140 004840 006110 011170 001270 049770 017670 006380 010580 006090 011200 025820 025000 004410 017800 002820 005850 004380 004255 007540 003090 000140 005250 005830 007120 004990 006125 005190 007690 010100 021820 006980 011780 009540 002100 010145 000520 004310 009290 003080 011390 010690 003780 000070 006805 000155 012800 003920 002300 020000 005935 047050 064960 002000 008490 007810 012205 027390 001940 005720 009155 001685 068290 017810 005500 014820 012320 010660 015890 027970 002070 003200 001680 003535 003490 000650 016800 003030 001770 012330 006390 005300 001440 009415 021050 034300 028050 009190 005750 001430 012280 010950 036460 005180 007570 000225 024090 002995 001630 002210 000725 004540 002390 002020 004830 002870 003120 033270 025860 018500 051630 002025 001755 012030 001390 019170 018670 003220 051910 003650 001530 005110 003350 069620 001725 004450 004987 010955 058650 014915 003620 008775 003460 013580 000850 000210 017940 014910 001040 004270 000040 002380 001820 011230 000227 012170 002320 000815 021240 015590 009810 003610 009200 029530 005010 001470 014580 024070 023150 004700 009275 042670 003570 002410 004770 000950 005320 001250 063160 005965 000400 004430 000590 004130 001340 010050 014530 004200 002990 005360 009460 001210 002450 001550 009140 008500 009320 012510 025890 012750 024890 007210 012450 001065 016610 000300 025620 010140 002880 039130 029460 027740 023590 001130 011790 020120 000080 017390 003560 007110 004565 005610 002250 069460 000480 005745 000105 005420 031820 006280 001360 001380
# 008000 005390 005745 005610 009415 069260 015260 008040 004565 014910 017550 000480 014160 003495 000105 005440 005110 001800 009070 010050 006110 001360 000227 004545 003690 007190 005420 001465 005300 002450 016450 009140 003160 021820 005870 005740 007630 011200 008350 001527 001525 001450 000860 001067 064960 001430 011150 037710 029530 011160 006570 003220 002360 006405 019180 004020 003419 007340 006390 016580 000145 001770 010060 019490 008970 000325 004870 005070 009160 002350 001820 002390 001550 051905 004770 010660 004170 012800 000890 002780 014915 002170 001270 005850 004415 007690 003475 005935 014280 009680 001620 000670 001045 004800 005965 002920 006220 000885 000225 009275 005387 011700 012205 003540 006800 003530 002070 002550 069960 005620 000910 002600 004920 004255 068875 004987 000810 003555 004970 033250 009155 105560 001750 000725 003350 024090 033530 012030 004540 000370 035000 015020 007590 000660 014680 006360 010950 004490 002720 001080 000155 005960 005500 003650 004430 025560 030610 001230 009470 010955 068870 010620 020000 004450 001685 001755 003850 004980 006400 005450 003535 001530 005810 016380 001380 004080 003520 001520 005430 003060 008110 003010 005980 005880 175330 001420 002250 006660 002630 010580 007120 001120 010770 010130 004910 001065 047040 000500 016360 024070 008490 007700 006340 024900 058430 002030 139130 009310 005680 002200 009830 033240 008870 013520 010690 025000 017800 009290 033180 007980 002790 005930 002960 004370 004060 008600 006490 017370 018880 011230 023450 071050 003000 013700 004200 025530 012750 012280 066575 005800 003580 002460 001725 012600 025540 030000 000760 000815 034300 012200 017040 005389 001060 004270 005945 003240 006120 002270 004990 051630 000070 047050 004090 011090 017900 004960 003120 025890 012690 005725 003280 011810 004250 005305 008560 007610 021960 014990 001510 005190 000650 010040 069620 008260 004985 042670 001940 004380 000950 007810 000490 009835 011280 027970 018500 016610 000720 011000 005320 000320 011690 016710 009450 001440 005380 000640 003470 011760 001515 000880 004310 002995 008930 030720 034020 000590 007310 031440 036460 003460 001070 011930 003300 024720 000240 002005 017670 004720 009970 006980 138930 014580 004130 003030 036530 005830 025860 009200 035150 006650 086790 005940 011155 005030 003410 005385 007540 016590 000430 002210 003680 006370 009420 009150 004710 021050 001275 014440 007575 006200 009180 001200 005720 032640 007110 027390 000270 002820 005010 000390 002310 000850 000215 015230 000040 016880 003720 025820 021240 017810 000180 010140 001740 000230 005257 005820 014790 018670 000300 015590 003545 003610 001470 010600 001720 002900 023590 002380 023960 027740 003230 016800 000400 002240 009540 009440 007815 003070 002320 012610 004410 002420 006125 005750 003080 018470 008730 000540 004150 003560 002140 004000 001260 007160 002795 010100 020760 001570 015350 000210 001340 002620 002100 004140 001210 003415 003550 006260 001790 036580 039130 016090 067830 028050 002025 029460 000020 012450 008250 006805 023800 005950 012330 011780 009810 010145 006090 002760 015890 003200 014285 015760 012170 012510 002020 008770 017960 000157 013570 001040 001880 000990 006840 003480 000120 000060 001745 012320 015860 055550 003490 030210 005490 004840 005690 010780 049770 012630 051900 051910 066570 015360 011300 016385 005360 011790 017390 007860 014820 002840 014130 002355 000970 008060 008775 004560 003780 003075 017940 030200 010820 004690 000100 008420 033270 002000 001500 014530 002810 068290 008700 001630 002700 020120 025620 026890 004365 011170 000075 001390 003090 009580 004835 063160 006740 000140 000220 033780 005180 000700 000520 019170 000050 001680 003620 051915 005090 000150 009270 007460 001140 009410 031820 033920 011785 007570 006380 001250 008355 006280 002300 009240 002990 058650 001130 003570 005250 004135 004100 003920 000545 023150 030790 002870 001290 004830 010120 003830 011390 005255 009460 002410 004700 013580 001795 014825 019175 004989 001460 007210 006060 004105 009190 006345 009320 024890 003960 000547 069460 011420 008500 013000 003465 002710 003925 002880 000080
# 051630 001530 003925 008000 014160 000650 006090 020120 008500 000480 005110 066570 000545 000860 002790 008350 005380 015260 066575 004100 002140 002710 006200 000157 008775 030000 005387 036460 001250 033250 004960 001520 000520 001340 005930 001515 011150 001230 001440 000270 003650 000220 018470 002700 007630 016880 010950 064960 051900 023590 003690 026890 005389 003410 004989 003620 013580 003780 000540 014130 005950 010040 011690 000070 004700 007980 005935 007160 006380 005385 025860 015860 004910 010145 003560 019175 002100 004060 034020 001620 012690 002795 005010 005800 009150 005810 005255 007190 012200 010955 012510 017960 013520 001380 004170 010130 016590 003540 009155 006260 017040 063160 008250 004380 003350 000390 005090 003419 007810 003075 014910 035150 037710 004105 024070 007700 004080 009680 007310 001940 009440 001880 004255 105560 004980 021240 000660 012030 011390 003850 013000 009270 014530 015230 011300 006660 030790 003830 004090 025560 001130 017900 009410 002020 000215 029530 004840 010100 009810 042670 003120 000990 002460 001770 025530 004020 000500 003160 008420 014790 001680 000060 011230 069460 010820 001460 000100 005965 011930 068870 004135 027390 006125 028050 010600 003030 001500 001720 005725 004540 069960 012630 005750 005820 055550 000810 018500 000370 002070 013700 004990 009450 086790 015590 003465 001570 025000 012750 029460 003550 023960 003060 003545 071050 004720 020760 012320 003470 005940 001040 025620 006370 005870 006805 007610 002000 017370 015890 003230 007570 004130 001630 004800 007110 003480 002210 032640 000590 011780 017810 139130 012450 006840 017670 004200 006345 007860 004987 008040 033240 005830 001430 002240 006400 003280 001820 068875 015020 006800 049770 003580 000720 000910 007815 067830 001060 016450 011200 000180 004410 002350 003070 003000 024090 058430 004710 011000 000140 005880 009310 011155 005430 002900 015760 010770 010620 017800 020000 006980 047040 010690 001685 001510 025820 000547 010050 010060 001450 001420 016385 019180 001045 051905 024890 003720 021960 030210 006490 005500 003535 009420 006405 001740 030200 006650 012280 025890 007540 001210 016380 000227 000815 016360 034300 000640 000950 002380 000700 018880 009970 138930 005320 014820 175330 004450 002820 000760 027970 000155 013570 047050 030610 000040 003010 004920 003460 000050 009200 011280 002870 011420 023150 005250 005680 014915 003555 002995 021820 003080 031440 006340 005945 008730 002550 001725 011160 004000 001070 010660 024900 008770 000240 000210 001527 004415 009180 069260 018670 051915 008930 008870 001795 000850 008110 003240 008970 014680 006360 001120 000020 003490 012170 006570 001275 019490 001140 008700 002025 068290 008560 007210 001200 024720 009290 000105 033180 000725 004985 000430 002960 002310 014990 001790 016610 023450 004545 004140 005257 019170 002200 016090 005450 006740 000490 003090 005690 002250 002390 002420 000880 003495 002320 001067 003530 005300 004310 002360 001065 004870 000225 021050 000320 004560 007340 000970 016580 001080 033530 003415 002840 011810 011790 002620 015350 015360 002170 000075 001390 012330 005490 002920 006220 036530 039130 003475 001550 069620 000080 011785 008060 004970 009580 017940 001745 002630 030720 000230 005190 009830 014285 010580 002810 002005 004770 014440 000400 004365 009070 010140 033270 004370 005360 002030 002990 009540 001525 005030 023800 001270 006060 008355 002355 001750 002600 014825 009470 011760 025540 058650 004690 003300 005960 003960 001800 000120 002270 011700 008600 004490 003220 009460 006120 003680 001290 008490 012610 005620 005180 000150 003610 000885 036580 012205 005720 003570 005305 007460 014280 003520 007120 009320 007590 009835 002300 000325 004430 002760 002720 001755 016800 006110 012800 004250 000300 033780 005850 005980 035000 012600 014580 004270 007690 005440 010120 009140 000670 005420 003200 009160 051910 000145 005610 001470 027740 008260 017390 001260 004150 002780 009275 001360 000890 009190 010780 006280 007575 001465 004835 016710 031820 005740 017550 005070 003920 009240 005390 004565 011170 011090 009415 002450 004830 006390 005745 033920 002410 002880
# 017810 011280 023150 004060 003160 009270 002880 018470 005070 009460 034020 014160 001430 000500 002410 000725 013580 009180 019180 005440 008350 009190 002200 001250 004960 012030 002250 000480 014910 004270 001360 005745 011790 014285 013000 003960 006370 002450 012510 005870 001510 003410 010130 033920 017040 000890 001770 009160 000270 000180 004565 068870 009310 002355 011230 011930 007110 001260 034300 004700 010660 000105 001465 015860 009275 001740 012280 012170 004840 020760 017370 005750 001380 010120 005380 011090 004380 001570 009240 019170 000145 021240 003560 000590 006280 042670 020000 031820 005385 032640 012630 008970 024720 005110 021960 011155 051900 007120 000720 003490 010820 138930 007815 007610 002720 016880 005450 064960 002870 004415 000700 018500 105560 028050 069460 005420 012205 005830 005950 001120 069960 003415 004720 025560 005010 004980 019490 009200 003850 030210 001630 003090 007575 003010 001440 003075 006800 035000 033270 029460 002420 015890 001060 016360 003540 017900 003070 002990 006400 004430 068875 000040 001080 005690 025620 004800 014580 007570 007190 001040 005940 036580 000070 009290 003350 139130 007340 005945 036530 009450 012600 016590 036460 007980 071050 007310 030790 030610 009150 014915 000660 000520 008260 007690 175330 005190 001470 012800 010040 003030 011300 009420 004020 004970 003650 014820 017550 055550 008770 005389 005960 014280 010140 014990 001065 015590 002460 003680 086790 015020 011200 024890 066570 005800 017960 016580 001525 010950 011780 006405 004130 004990 004255 000120 004690 063160 005387 024070 002020 016710 005180 001680 012200 000815 009580 000540 011420 004830 000150 006805 006380 010690 008560 010600 023960 013570 019175 004560 000155 004490 030720 000640 008110 001620 000220 047040 008700 004985 006740 006220 008600 001140 014680 010770 003280 004910 035150 003620 007700 014825 006090 012690 000210 001795 003690 003545 003200 016380 011150 002170 049770 008870 004450 000390 001685 007630 025540 003475 000850 004000 010620 004987 027390 012610 001520 047050 011690 005390 033530 007860 018670 037710 001130 011000 011160 004920 009540 006260 015350 005610 016610 002900 066575 003920 002790 004410 069620 008250 002810 007210 009810 003780 000490 003480 002820 004135 001515 017670 002070 000400 001045 003000 002700 016450 008930 000810 005360 027970 002995 013700 025530 021050 009470 068290 006840 005810 051910 001200 000225 001940 004140 006650 002310 001745 001450 000240 009140 003060 002380 009970 013520 014130 002000 001420 005980 002600 005930 006360 011700 011810 009320 008000 002920 006570 023450 003120 030200 024900 001275 008490 000100 015760 014440 009155 003720 001270 010060 004105 005720 011760 007810 002630 001790 005320 002620 027740 003555 000320 003300 004310 005430 002005 001527 009680 000157 004250 003550 003240 026890 001230 004365 000230 006200 002550 025890 005500 025820 000020 033240 002350 007460 002360 006120 008775 004545 011785 012330 017800 000885 006125 017940 003520 000990 000970 024090 002710 005740 015360 008040 002025 000050 006060 000060 000140 001290 004870 000910 002320 000670 000950 005935 033780 000545 006340 003610 025000 003470 067830 012750 010580 014530 007590 005300 000430 009440 005680 008060 001070 008730 000227 010100 005305 011390 033180 005820 003460 001750 005250 004540 006490 025860 004835 009415 005255 010955 001500 004989 000880 001820 002840 004200 058650 030000 003465 000760 005090 002270 007540 002760 000370 016800 002240 015230 006345 000215 005850 020120 005030 012320 003535 004090 001390 006980 002210 003580 003495 023590 002795 004170 004770 001880 009070 016385 000325 001720 009410 021820 006390 014790 003220 002960 001460 002030 029530 005257 002100 008355 002390 010050 005725 004370 005490 005965 008420 006660 010780 003570 031440 002780 004710 015260 001340 023800 002140 058430 004100 004080 003830 016090 008500 018880 033250 003530 003080 051905 012450 039130 002300 003419 005620 017390 001725 001800 003230 005880 001550 010145 051630 006110 007160 009835 051915 003925 001210 000300 000860 000075 004150 000547 001067 001755 011170 001530 009830 000650 069260""".split()
#
#
#
#
# # vocab = set(test_sentence)
# #set()函数创建一个无须不重复的函数集
#
# # word_to_ix = {word: i for i, word in enumerate(vocab)}
# #制作word_to_ix字典
# # print(word_to_ix)
# # print(111111)
# # word_idxs = [word_to_ix[w] for w in vocab]
# # print(vocab)
# # print(111111)
# # print(word_idxs)
#
# #print(test_sentence)
#
# #print(get_target(test_sentence,10,2))
# #for inputs, targets in get_batches(test_sentence, 10):
# #    print(inputs,targets)
# #print(get_batches(test_sentence,10,5))
# #print(1)



# 2/19
import torch

x = torch.FloatTensor(torch.rand([2]))
print('x', x)
y = torch.FloatTensor(torch.rand([2]))
print('y', y)

similarity = torch.cosine_similarity(x, y, dim=0)
print('similarity', similarity)