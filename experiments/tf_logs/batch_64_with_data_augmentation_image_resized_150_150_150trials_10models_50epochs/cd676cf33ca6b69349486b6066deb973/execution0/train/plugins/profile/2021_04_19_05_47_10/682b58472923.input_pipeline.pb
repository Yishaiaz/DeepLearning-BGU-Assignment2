	t??%`j@t??%`j@!t??%`j@	?B|j?????B|j????!?B|j????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6t??%`j@??A?^@1????P?U@A?#ӡ????I???o_??Y?̰Q?o??*	????+B?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2<?8b-^@!{C????X@)X?vMH{@1
??9?mW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??E??\??!G??+?@).W?6ɏ??1?(??#??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?0&????!?y??b@)?}r 
??1??k???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???x???!?e&9v???)???x???1?e&9v???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@ȳ˷>??!gDw?4???)ȳ˷>??1gDw?4???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\?????!??r9&P??)L6l?ۗ?1?j1?>??:Preprocessing2F
Iterator::Model?xy:W???!e!??ß??)MI???*}?1???|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?B|j????IY???~?L@Q!w????D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??A?^@??A?^@!??A?^@      ??!       "	????P?U@????P?U@!????P?U@*      ??!       2	?#ӡ?????#ӡ????!?#ӡ????:	???o_?????o_??!???o_??B      ??!       J	?̰Q?o???̰Q?o??!?̰Q?o??R      ??!       Z	?̰Q?o???̰Q?o??!?̰Q?o??b      ??!       JGPUY?B|j????b qY???~?L@y!w????D@