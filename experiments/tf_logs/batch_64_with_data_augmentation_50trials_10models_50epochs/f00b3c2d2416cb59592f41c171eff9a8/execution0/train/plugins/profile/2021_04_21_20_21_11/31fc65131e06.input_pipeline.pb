	??k?Sf@??k?Sf@!??k?Sf@	??!?n;????!?n;??!??!?n;??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??k?Sf@???"?
@1??*l??e@I??????@Yo??\???*	??ʡEx?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV21?q??'@!?o?Q??X@)?sCSf'@12 ??TWX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?Ƅ?K???!??[?!???)ywd?6???1	%<?^???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??RB????!?=?j??))??Pj/??1??d????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@Lǜg?K??!??-?????)Lǜg?K??1??-?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch3????!?E??ݿ??)3????1?E??ݿ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism6?????!?0-Px???)uF^?Ē?1?7\?%???:Preprocessing2F
Iterator::Model_??W???!??6W???)????c{?1?i?}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??!?n;??I??Y??}	@QO?i?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???"?
@???"?
@!???"?
@      ??!       "	??*l??e@??*l??e@!??*l??e@*      ??!       2      ??!       :	??????@??????@!??????@B      ??!       J	o??\???o??\???!o??\???R      ??!       Z	o??\???o??\???!o??\???b      ??!       JGPUY??!?n;??b q??Y??}	@yO?i?X@