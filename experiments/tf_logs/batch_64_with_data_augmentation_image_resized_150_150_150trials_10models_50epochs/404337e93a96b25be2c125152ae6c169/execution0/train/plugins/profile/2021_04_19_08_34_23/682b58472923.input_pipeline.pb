	??hq?f@??hq?f@!??hq?f@	Wv?4???Wv?4???!Wv?4???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??hq?f@D?R?V@1ӆ???wU@A??? ?r??I?nض(???Y??v????*	????&Գ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2P9&???@!?ۓ???X@)p	????@1??.c2^W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??b?dU??!??R?T@)?8?Z????1????W@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@L??1%??!X,ؽ??@) C?*q??1y_'S?y??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??4*p???!6??(>H??)??4*p???16??(>H??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchb0?̕??!v?m???)b0?̕??1v?m???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismݔ?Z	ݱ?!t??????)???`?H??1???aP???:Preprocessing2F
Iterator::Model?q ????!t	???)?? Z+z?1Hp
H??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Xv?4???I?M?Y^I@QGھӟGH@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	D?R?V@D?R?V@!D?R?V@      ??!       "	ӆ???wU@ӆ???wU@!ӆ???wU@*      ??!       2	??? ?r????? ?r??!??? ?r??:	?nض(????nض(???!?nض(???B      ??!       J	??v??????v????!??v????R      ??!       Z	??v??????v????!??v????b      ??!       JGPUYXv?4???b q?M?Y^I@yGھӟGH@