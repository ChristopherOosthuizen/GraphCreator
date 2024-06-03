from deepeval import assert_test
import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

import src.GraphCreation.textformatting as tx


input_one = open("src/prompts/formatting").read()+" "+"""<li id="coll-download-as-rl" class="mw-list-item"><a href="/w/index.php?title=Special:DownloadAsPdf&amp;page=Knight_of_the_shire&amp;action=show-download-screen" title="Download this page as a PDF file"><span>Download as PDF</span></a></li><li id="t-print" class="mw-list-item"><a href="/w/index.php?title=Knight_of_the_shire&amp;printable=yes" title="Printable version of this page [ctrl-option-p]" accesskey="p"><span>Printable version</span></a></li>
		</ul>
		
	</div>
</div>

</div>

									</div>
				
	</div>
</div>

							</nav>
						</div>
					</div>
				</div>
				<div class="vector-column-end">
					<div class="vector-sticky-pinned-container">
						<nav class="vector-page-tools-landmark" aria-label="Page tools">
							<div id="vector-page-tools-pinned-container" class="vector-pinned-container">
				
							</div>
		</nav>
						<nav class="vector-appearance-landmark" aria-label="Appearance">
						</nav>
					</div>
				</div>
				<div id="bodyContent" class="vector-body ve-init-mw-desktopArticleTarget-targetContainer" aria-labelledby="firstHeading" data-mw-ve-target-container="">
					<div class="vector-body-before-content">
							<div class="mw-indicators">
		</div>

						<div id="siteSub" class="noprint">From Wikipedia, the free encyclopedia</div>
					</div>
					<div id="contentSub"><div id="mw-content-subtitle"></div></div>
					
					
					<div id="mw-content-text" class="mw-body-content"><div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr"><div class="shortdescription nomobile noexcerpt noprint searchaux" style="display:none">Formal title of MPs from county constituencies</div>
<p>

<b>Knight of the shire</b> (<a href="/wiki/Latin_language" class="mw-redirect" title="Latin language">Latin</a>: <i lang="la">milites comitatus</i>)<sup id="cite_ref-Tomlins1835_1-0" class="reference"><a href="#cite_note-Tomlins1835-1">[1]</a></sup> was the formal title for a <a href="/wiki/Member_of_parliament" title="Member of parliament">member of parliament</a> (MP) representing a <a href="/wiki/County_constituency" class="mw-redirect" title="County constituency">county constituency</a> in the <a href="/wiki/House_of_Commons_of_the_United_Kingdom" title="House of Commons of the United Kingdom">British House of Commons</a>,  from its origins in the medieval <a href="/wiki/Parliament_of_England" title="Parliament of England">Parliament of England</a> until the <a href="/wiki/Redistribution_of_Seats_Act_1885" title="Redistribution of Seats Act 1885">Redistribution of Seats Act 1885</a> ended the practice of each <a href="/wiki/Counties_of_the_United_Kingdom" title="Counties of the United Kingdom">county</a> (or <i><a href="/wiki/Shire" title="Shire">shire</a></i>) forming a single <a href="/wiki/Constituency" class="mw-redirect" title="Constituency">constituency</a>. The corresponding titles for other MPs were <i><a href="/wiki/Burgess_(title)" title="Burgess (title)">burgess</a></i> in a <a href="/wiki/Borough_constituency" class="mw-redirect" title="Borough constituency">borough constituency</a>  (or <i><a href="/wiki/Citizen" class="mw-redirect" title="Citizen">citizen</a></i> if the borough had <a href="/wiki/City_status_in_the_United_Kingdom" title="City status in the United Kingdom">city status</a>) and <i><a href="/wiki/Barons_of_the_Cinque_Ports" class="mw-redirect" title="Barons of the Cinque Ports">baron</a></i> for a <a href="/wiki/Cinque_ports_parliament_constituencies" title="Cinque ports parliament constituencies">Cinque Ports constituency</a>. Knights of the shire had more prestige than burgesses, and sitting burgesses often stood for election for the shire in the hope of increasing their standing in Parliament.
</p><p>The name "knight of the shire" originally implied that the representative had to be a <a href="/wiki/Knight" title="Knight">knight</a>, and the <a href="/wiki/Writ_of_election" title="Writ of election">writ of election</a> referred to a <a href="https://en.wiktionary.org/wiki/belted_knight" class="extiw" title="wikt:belted knight">belted knight</a> until the 19th century;<sup id="cite_ref-Tomlins1835_1-1" class="reference"><a href="#cite_note-Tomlins1835-1">[1]</a></sup> but by the 14th century men who were not knights were commonly elected.<sup id="cite_ref-2" class="reference"><a href="#cite_note-2">[2]</a></sup> An act of <a href="/wiki/Henry_VI_of_England" title="Henry VI of England">Henry VI</a> (<a href="/wiki/23_Hen._6" class="mw-redirect" title="23 Hen. 6">23 Hen. 6</a>. c. 14) stipulated that those eligible for election were knights and "such notable <a href="/wiki/Esquire" title="Esquire">esquires</a> and <a href="/wiki/Gentlemen" class="mw-redirect" title="Gentlemen">gentlemen</a> as have <a href="/wiki/Estate_(law)" title="Estate (law)">estates</a> sufficient to be knights, and by no means of the degree of <a href="/wiki/Yeoman" title="Yeoman">yeoman</a>".<sup id="cite_ref-3" class="reference"><a href="#cite_note-3">[3]</a></sup>
</p>"""
def test_case():
    presion_metric = ContextualPrecisionMetric()
    test_case = LLMTestCase(
        
        input_text=input_one,
        expected_output="From Wikipedia, the free encyclopedia\n\nFormal title of MPs from county constituencies\n\nKnight of the shire (Latin: milites comitatus)[1] was the formal title for a member of parliament (MP) representing a county constituency in the British House of Commons, from its origins in the medieval Parliament of England until the Redistribution of Seats Act 1885 ended the practice of each county (or shire) forming a single constituency. The corresponding titles for other MPs were burgess in a borough constituency (or citizen if the borough had city status) and baron for a Cinque Ports constituency. Knights of the shire had more prestige than burgesses, and sitting burgesses often stood for election for the shire in the hope of increasing their standing in Parliament.\nThe name \"knight of the shire\" originally implied that the representative had to be a knight, and the writ of election referred to a belted knight until the 19th century;[1] but by the 14th century but men who were not knights were commonly elected. An act of Henry VI (23 Hen. 6. c. 14) stipulated that those eligible for election were knights and \"such notable esquires and gentlemen as have estates sufficient to be knights, and by no means of the degree of yeoman\".[3]",
        actual_output=tx.format_text(input_one, ""),
    )
    assert_test([test_case], [presion_metric])