	?7k?g@?7k?g@!?7k?g@	?}??????}?????!?}?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?7k?g@?p]1\Y@1?]?\U@Ap\?M4??I??mnL@YH?V
??*	????.?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?c???H@!KR/?8?X@)????rK@1?????9W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@i??֦???!????H@)???Y???1LZtFx?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@ٯ;?y???!???̊@)?Ɍ??^??1?d?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??^f??!;?????)??^f??1;?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?W歺??!Nu:I҂??)?W歺??1Nu:I҂??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???A_z??!?5øP(??)s,??̓?1#?????:Preprocessing2F
Iterator::ModelB?"LQ.??!Ym+??q??)?t?? ?{?1L?AۇK??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?}?????I??&??BK@QK??\?yF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p]1\Y@?p]1\Y@!?p]1\Y@      ??!       "	?]?\U@?]?\U@!?]?\U@*      ??!       2	p\?M4??p\?M4??!p\?M4??:	??mnL@??mnL@!??mnL@B      ??!       J	H?V
??H?V
??!H?V
??R      ??!       Z	H?V
??H?V
??!H?V
??b      ??!       JGPUY?}?????b q??&??BK@yK??\?yF@