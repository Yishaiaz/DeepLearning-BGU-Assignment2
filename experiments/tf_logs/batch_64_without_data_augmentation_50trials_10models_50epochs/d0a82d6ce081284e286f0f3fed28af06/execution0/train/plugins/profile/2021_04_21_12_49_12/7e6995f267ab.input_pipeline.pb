	B!??t@B!??t@!B!??t@	?p?%?"???p?%?"??!?p?%?"??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-B!??t@!??q4@1O?6???t@I?Z(?|@Y?N#-????*	?"???_?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?4ӽN?*@!gv)1??X@)?i?{??)@1??6X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@Eg?E(???!:Ob߹@)??=Զ??1?O?<???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@1`?U,~??!4??N1??)1`?U,~??14??N1??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?7/N|???!E$ևZ??)?????n??1#?x??Q??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9?Վ???!?6?z????)9?Վ???1?6?z????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/M??.??!P?ֈ??)?o?DIH??1??HW???:Preprocessing2F
Iterator::ModelY?n?ͷ?!<?Dk???)?27߈?y?1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?p?%?"??I?J??????Q???T??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!??q4@!??q4@!!??q4@      ??!       "	O?6???t@O?6???t@!O?6???t@*      ??!       2      ??!       :	?Z(?|@?Z(?|@!?Z(?|@B      ??!       J	?N#-?????N#-????!?N#-????R      ??!       Z	?N#-?????N#-????!?N#-????b      ??!       JGPUY?p?%?"??b q?J??????y???T??X@