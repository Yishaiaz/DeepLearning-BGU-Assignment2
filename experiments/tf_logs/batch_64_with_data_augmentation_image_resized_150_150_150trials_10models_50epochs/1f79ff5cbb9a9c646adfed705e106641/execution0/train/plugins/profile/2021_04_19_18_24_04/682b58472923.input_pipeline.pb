	7erl@7erl@!7erl@	ƪE??"??ƪE??"??!ƪE??"??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails67erl@&r??a@1??Z}?U@A??im۳?IRew?@Y?F<????*	+???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2X??G@!?8s??X@)?R?r/p@1?????~V@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@u><K???!?Fgy?@)/?????1U?Z)?@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@^-wf????!8?d?e @)??u6????1?R^?n?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@u??<???!?+??w??)u??<???1?+??w??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?XP?i??!ѻw?)??)?XP?i??1ѻw?)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????6???!I??????)??M~?N??1??P.07??:Preprocessing2F
Iterator::Model???:8ط?!`?1c~??)[?a/?}?1?p/iC??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ƪE??"??I?K?$??N@Q?(MD?AC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&r??a@&r??a@!&r??a@      ??!       "	??Z}?U@??Z}?U@!??Z}?U@*      ??!       2	??im۳???im۳?!??im۳?:	Rew?@Rew?@!Rew?@B      ??!       J	?F<?????F<????!?F<????R      ??!       Z	?F<?????F<????!?F<????b      ??!       JGPUYƪE??"??b q?K?$??N@y?(MD?AC@