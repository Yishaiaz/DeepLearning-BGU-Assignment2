	?V{?K3u@?V{?K3u@!?V{?K3u@	'????!??'????!??!'????!??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?V{?K3u@??b???@1(+????t@I??wc???Y[`???f??*	I?z?E?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2NF?aܥ&@!Z1-?8?X@)^h??H&@1???&!X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?ۼqR??!?N???F@)?HP???1?X?[????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@eo)????!E`?(???)?_"?:???1%??.5??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@j?t???!????"??)j?t???1????"??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch3?????!?V'Yy???)3?????1?V'Yy???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?b?: ???!??D????)e8?πz??1y?þ?Y??:Preprocessing2F
Iterator::Model?????l??!<Sg??c??)??I~įx?1ȣ+?L??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9'????!??I??Lk[???Q&?]???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??b???@??b???@!??b???@      ??!       "	(+????t@(+????t@!(+????t@*      ??!       2      ??!       :	??wc?????wc???!??wc???B      ??!       J	[`???f??[`???f??![`???f??R      ??!       Z	[`???f??[`???f??![`???f??b      ??!       JGPUY'????!??b q??Lk[???y&?]???X@