import re
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime

def elapsed_time_since_modified(file_path):
    """Returns the elapsed time since the file was last modified."""
    try:
        modification_time = os.path.getctime(file_path)
        current_time = datetime.datetime.now().timestamp()
        #current_time = os.path.getmtime(file_path)
        elapsed_seconds = current_time - modification_time
        return elapsed_seconds #datetime.timedelta(seconds=elapsed_seconds)
    except FileNotFoundError:
        return None



def extract_losses_from_log(log_file_path):
    log_content = ''
    #if log_file_path.endswith('.2Xdata.4Xb'):
    #    with open(log_file_path[:-len('.2Xdata.4Xb')], 'r') as file:
    #        log_content += file.read()
    #if log_file_path.endswith('.cont'):
    #    with open(log_file_path[:-len('.cont')], 'r') as file:
    #        log_content += file.read()
    #if 'all.fixrelu.simplifyinput' in log_file_path:
    #    with open(log_file_path.replace('.fixrelu.simplifyinput', ''), 'r') as file:
    #        log_content += file.read()
    #    with open(log_file_path.replace('.simplifyinput', ''), 'r') as file:
    #        log_content += file.read()
    #elif 'all.fixrelu' in log_file_path:
    #    with open(log_file_path.replace('.fixrelu', ''), 'r') as file:
    #        log_content += file.read()
    #elif 'all' in log_file_path:
    #    pass
    #elif '.loadbest.context8.contfreezernn' in log_file_path:
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.freezernn.context8', 'r') as file:
    #        log_content += file.read()
    #elif '.loadbest.freezernn.context8' in log_file_path:
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest', 'r') as file:
    #        log_content += file.read()
    #elif '.loadbest.context8' in log_file_path:
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest', 'r') as file:
    #        log_content += file.read()
    #elif '.loadbest' in log_file_path:
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered', 'r') as file:
    #        log_content += file.read()
    #    with open('log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont', 'r') as file:
    #        log_content += file.read()
    #elif '.cont3' in log_file_path:
    #    with open(log_file_path.replace('.cont3', ''), 'r') as file:
    #        log_content += file.read()
    #    with open(log_file_path.replace('.cont3', '.cont'), 'r') as file:
    #        log_content += file.read()
    #    with open(log_file_path.replace('.cont3', '.cont2'), 'r') as file:
    #        log_content += file.read()
    #elif '.cont2' in log_file_path:
    #    with open(log_file_path.replace('.cont2', ''), 'r') as file:
    #        log_content += file.read()
    #    with open(log_file_path.replace('.cont2', '.cont'), 'r') as file:
    #        log_content += file.read()
    #elif '.cont' in log_file_path:
    #    with open(log_file_path.replace('.cont', ''), 'r') as file:
    #        log_content += file.read()
    with open(log_file_path, 'r') as file:
        log_content += file.read()

    # Extract losses using regex (e.g., loss=1.88e+03)
    loss_pattern = re.compile(r'loss_step_step=([\d\.eE+-]+)')
    losses = [math.log(float(loss)) for loss in loss_pattern.findall(log_content)]
    if len(losses) == 0:
        loss_pattern = re.compile(r'loss_step=([\d\.eE+-]+)')
        losses = [math.log(float(loss)) for loss in loss_pattern.findall(log_content)]
    if len(losses) == 0:
        loss_pattern = re.compile(r'loss_part1_step=([\d\.eE+-]+)')
        losses = [math.log(float(loss)) for loss in loss_pattern.findall(log_content)]
    if len(losses) == 0:
        loss_pattern = re.compile(r'loss__step=([\d\.eE+-]+)')
        losses = [math.log(float(loss)) for loss in loss_pattern.findall(log_content)]
    print (len(losses))
    
    return losses

def smooth_losses(losses, window_size=100):
    import pandas as pd
    series = pd.Series(losses)
    return series.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    #if len(losses) < window_size:
    #    window_size = max(1, len(losses) // 10)
    #return np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

def plot_and_compare_losses(losses_dict, output_png='psearch_newdatajdiffusion_loss_curve_comparison_computecanada.png'):
    """
    Plot and compare multiple loss curves on a single graph.
    
    :param losses_dict: A dictionary where keys are labels and values are tuples of (losses, smoothed_losses).
    :param output_png: The filename for the output plot image.
    """
    plt.figure(figsize=(12, 6))
    
    for label, (losses, smoothed_losses) in losses_dict.items():
        #plt.plot(range(len(losses)), losses, alpha=0.3, label=f'{label} Raw Loss')
        plt.plot(range(len(smoothed_losses)), smoothed_losses, linewidth=2, label=f'{label} Smoothed Loss')
        #plt.plot(range(len(losses)), losses, linewidth=2, label=f'{label} Smoothed Loss')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve Comparison')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_png)
    print(f'Loss curve comparison saved as {output_png}')
#

if __name__ == "__main__":
    log_file_paths = [
        #('a_hs4096_oc32_nl48_ar_cm1_2_mc320', 'log.a_hs4096_oc32_nl48_ar_cm1_2_mc320'),
        #('a_hs4096_oc32_nl48_ar_cm1_2_mc384', 'log.a_hs4096_oc32_nl48_ar_cm1_2_mc384'),
        #('a_hs4096_oc32_nl48_ar2_cm1_2_mc320', 'log.a_hs4096_oc32_nl48_ar2_cm1_2_mc320'),
        #('a_hs4096_oc32_nl48_ar_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar_cm1_2_3_mc320'),
        #('a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar2_cm1_2_3_mc320'),
        #('a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320', 'log.a_hs4096_oc32_nl48_ar4_cm1_2_3_mc320'),
        #('a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192', 'log.a_hs1024_oc4_nl20_ar2_4_8_cm1_2_3_5_mc192'),
        #('a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320', 'log.a_hs4096_oc32_nl48_ar2_4_8_cm1_2_3_5_mc320'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr4e5_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr1.6e4_b64'),
        #('b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128', 'log.b_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b128'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr4e5_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr1.6e4_b64'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b100'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu1'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu2'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu4'),
        #('b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8', 'log.b_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8'),
        #('orig', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont'),
        #('final', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.cont'),
        #('final1', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg'),
        #('final2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.bsz50'),
        #('final20', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5'),
        #('final3', 'log.final_hs4096_oc32_nl48_ar2_cm1_2_mc384_lr8e5_b64_gpu8_filtered.largeimg.lr4e5'),
        #('largeimg', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest'),
        #('all', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput'),
        #('fullreinit', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.fixed'),
        #('addattn', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.addattn.fixed.cont'),
        #('reinitrnn', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitrnn.fixed'),
        #('reinitcnn', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitcnn.fixed'),
        #('reinitnone', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone.fixed'),
        #('reinitnonehp', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone.highprecision.fixed'),
        #('cheat', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.fullreinit.reinitnone_cheat.fixed'),
        #('working', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain22'),
        #('not working', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain228'),
        #('running', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2'),
        #('running2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2_nopretrain_posttrain'),
        #('orig', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2.real.context32'),
        #('final', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32'),
        #('final', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.2Xdata.4Xb'),
        #('final2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.3Xdata.4Xb'),
        #('final3', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont'),
        #('filter', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont.filtered'),
        #('filterall', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.cont.cont.cont.filtered.all'),
        #('freezernn', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered'),
        #('freezernn1Xb', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb'),
        #('unfreeze', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze'),
        #('unfreezefiltered', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.filtered'),
        #('freezernn1Xbchallenging', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.challenging'),
        #('freezernn1Xbafterchallenging', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging'),
        #('newdatapretrain', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging'),
        #('newdatapretrainaddc', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc'),
        #('newdatapretrainaddc_all', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrainreal.context32.4Xdata.4Xb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew'),
        #('newdatapretrainaddc_all_more_c', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c'),
        #('newdatapretrainaddc_all_more_c_alldata', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata'),
        #('diffusion_c', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c'),
        #('diffusion_c_alldatafrozen', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata'),
        #('diffusion_c_alldatafrozen_unfreeze', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss'),
        #('diffusion_c_alldatafrozen_unfreeze_4Xb', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb'),
        #('diffusion_c_alldatafrozen_unfreeze_4Xb_ss005', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005'),
        #('diffusion_c_alldatafrozen_unfreeze_4Xb_ss005_cont', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont'),
        #('diffusion_c_alldatafrozen_unfreeze_4Xb_ss005_cont_lr2e5', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5'),
        #('diffusion_c_alldatafrozen_unfreeze_4Xb_ss005_cont_lr2e5_ctx64', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64'),
        ('computecanada', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada'),
        ('lr8e5', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.lr8e5'),
        ('lr5e6', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.lr5e6'),
        ('challenging', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample'),
        ('pretrain', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.pretrainrnn'),
        ('pretrain_challenging', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn'),
        ('pretrain_challenging_balanced', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn.balanced'),
        ('pretrain_challenging_balanced_lr5e6', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn.balanced.lr5e6'),
        ('pretrain_challenging_balanced_lr1.25e6', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn.balanced.lr1.25e6'),
        #('pretrain_challenging_balanced_lr8e5', 'log.fb.diffusion.freezernn.contfiltered.1Xb.unfreeze.afterchallenging.newdata.pretrainchallenging.addc.allnew.more_c.alldata.diffusion_c.alldata.joint_noss.4Xb.ss005.cont.lr2e5.context64.computecanada.challengingandsample.pretrainrnn.balanced.lr8e5'),
        #('pretrain2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all.fixrelu.simplifyinput.debug.pretrain2282.samesetting2.real.contdebug'),
        #('pretrain2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8'),
        #('part1', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart1'),
        #('part2', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart2'),
        #('finetuneboth', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.finetunerealpart1synpart2'),
        #('pretrainboth', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512.pretrain2.context8.pretrainrealpart1synpart2'),
        #('prevall', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.all'),
        #('prev', 'log.standard_challenging_context32_nocond_fixnorm_all_scheduled_sampling_feedz_comb0.1_rnn_fixrnn_enablegrad_all_keyevent_cont_clusters_all_realall')
        #('largeimg_cont', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8'),
        #('largeimg_freeze', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.freezernn.context8'),
        #('largeimg_freezecont', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.loadbest.context8.contfreezernn'),
        #('1gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu1'),
        #('2gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu2'),
        #('4gpu', 'log.final_hs4096_oc32_nl48_ar_cm1_2_mc512_lr8e5_b64_gpu8_filtered.largeimg.lr4e5.gpu4'),
    ]
    
    losses_dict = {}
    for label, log_file_path in log_file_paths:
        losses = extract_losses_from_log(log_file_path)
        smoothed_losses = smooth_losses(losses, window_size=500)
        time_elapsed = elapsed_time_since_modified(log_file_path)
        if 'b128' in label:
            time_elapsed = 0.5
        elif 'b100' in label:
            time_elapsed = 64/100
        else:
            time_elapsed = 1
        print (label, len(losses), len(losses) / time_elapsed, smoothed_losses[-1])
        #losses_dict[label] = (losses[:24000], smoothed_losses[:24000])
        #losses_dict[label] = (losses[:256000], smoothed_losses[:256000])
        losses_dict[label] = (losses[:], smoothed_losses[:])
        #losses_dict[label] = (losses[:6000], smoothed_losses[:6000])
    
    plot_and_compare_losses(losses_dict)
