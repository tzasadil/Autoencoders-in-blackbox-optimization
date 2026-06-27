all_cols = ['vals', 'evals', 'pop_size', 'gen_mult', 'model', 'dim_red', 'instance','function', 'dim', 'elapsed_time', 'coco_directory', 'timestamp', 'budget', 'note', 'spearman_corr', 'spearman_pval', 'dists', 'selected_spread_ratio', 'selected_radius_ratio', 'selection_quality_gap', 'oracle_regret'] # and 'ranks', '
determining_cols = ['pop_size','gen_mult', 'model', 'dim_red','budget', 'note'] # the 'run settings' that we measure the performance of  
pure_desc_cols = ['pop_size','gen_mult','budget', 'note'] # the 'run settings' that we measure the performance of  

def get_full_desc(item):
    # nonempty_append = lambda name: ('_'+str(item[name]) if len(str(item[name]))>0 else '')
    if str(item['gen_mult']) == 1 or str(item['gen_mult']) == '1':
        desc = '_'.join([s for name in pure_desc_cols if len((s:=str(item[name])))>0])
    else:
        desc = '_'.join([s for name in determining_cols if len((s:=str(item[name])))>0])
    return desc

def get_storage_desc(item):
    dsc_cols = determining_cols + ['dim','instance','function']
    desc = '_'.join([s for name in dsc_cols if len((s:=str(item[name].iloc[0])))>0])
    return desc