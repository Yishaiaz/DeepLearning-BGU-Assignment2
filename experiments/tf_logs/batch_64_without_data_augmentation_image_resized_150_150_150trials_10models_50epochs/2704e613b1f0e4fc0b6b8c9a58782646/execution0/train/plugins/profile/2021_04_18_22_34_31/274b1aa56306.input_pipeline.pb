	̴?+??h@̴?+??h@!̴?+??h@	h2?aO??h2?aO??!h2?aO??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6̴?+??h@y??Z@1??Q?y?U@A?	j?֭?I??}q?J@Y???qť?*	??x?f?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??;??8@!s?2ǡ?X@)`;?O`@1?w#?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@T8?T???!??????@)??????1n?D@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@1?t?????!?_
P?|@)Ḍ?h??1?և?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@ù???!????@??)ù???1????@??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\?J???!????-???)\?J???1????-???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??)x
??!?d?>/???)eU???*??1B薯??:Preprocessing2F
Iterator::Modelc???&???!$cX3???)ŭ???w?1q?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9g2?aO??I?X?=?%L@Q-hA?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y??Z@y??Z@!y??Z@      ??!       "	??Q?y?U@??Q?y?U@!??Q?y?U@*      ??!       2	?	j?֭??	j?֭?!?	j?֭?:	??}q?J@??}q?J@!??}q?J@B      ??!       J	???qť????qť?!???qť?R      ??!       Z	???qť????qť?!???qť?b      ??!       JGPUYg2?aO??b q?X?=?%L@y-hA?E@