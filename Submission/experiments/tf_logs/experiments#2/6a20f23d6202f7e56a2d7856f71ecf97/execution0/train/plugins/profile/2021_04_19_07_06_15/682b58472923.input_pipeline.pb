	2??|Nh@2??|Nh@!2??|Nh@	??ӯYQ????ӯYQ??!??ӯYQ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails62??|Nh@?)??%YZ@1bg
?ׇU@A?>U?b??I?đ?@Y?<i????*	??n?=?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2C????p@!?=}???X@)M??StT@1???=?PW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@W_]????!Hl?\?o@)??Kǜg??1?????@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@l?<*???!?ٸ?p??)l?<*???1?ٸ?p??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@!;oc?#??!?????@)(
??<I??1*???l??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?h>?n??!OK??C??)?h>?n??1OK??C??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismp#e??ݰ?!?'"??W??)?XİØ??1Ul?????:Preprocessing2F
Iterator::Model???L??!4?? ???)qTn???v?1u??Z???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??ӯYQ??I???G?K@Q?Դ?%F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)??%YZ@?)??%YZ@!?)??%YZ@      ??!       "	bg
?ׇU@bg
?ׇU@!bg
?ׇU@*      ??!       2	?>U?b???>U?b??!?>U?b??:	?đ?@?đ?@!?đ?@B      ??!       J	?<i?????<i????!?<i????R      ??!       Z	?<i?????<i????!?<i????b      ??!       JGPUY??ӯYQ??b q???G?K@y?Դ?%F@