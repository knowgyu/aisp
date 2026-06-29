import{C as e,O as t,V as n,b as r,et as i,v as a,w as o,xt as s,y as c,yt as l}from"./modules/shiki-CkWyK9AO.js";import{nt as u,rt as d}from"./index-uBJohJK9.js";import{t as f}from"./slidev/default-o21k_nhW.js";import{t as p}from"./slidev/VClicks-r1EzxaUR.js";var m={__name:`recent-llm-architectures.md__slidev_283`,setup(m){let{$slidev:h,$nav:g,$clicksContext:_,$clicks:v,$page:y,$renderContext:b,$frontmatter:x}=d();return _.setup(),(d,m)=>{let h=p;return n(),c(f,s(t(l(u)(l(x),282))),{default:i(()=>[m[1]||=a(`h1`,null,`Convolutional Mixing`,-1),m[2]||=a(`img`,{src:`https://substack-post-media.s3.amazonaws.com/public/images/ad6e9cf3-2427-4aca-8afc-74202ddf93c4_2048x1208.png`,class:`mx-auto max-w-full max-h-77`},null,-1),m[3]||=a(`p`,null,[e(`Compressed Q, K에 `),a(`strong`,null,`sequence-mixing`),e(`과 `),a(`strong`,null,`channel-mixing`),e(`를 순차적으로 convolution 적용.`)],-1),r(`
Convolution(합성곱)은 입력 데이터 위를 작은 필터(커널)가 슬라이딩하면서 주변 값들을 가중합하는 연산입니다.

1D 예시 (시퀀스 기준):
커널 크기가 3이라면, 각 위치에서 자기 자신 + 양옆 값을 가중합해서 새 값을 만듭니다.

입력:   [A] [B] [C] [D] [E]
              ↓
커널:   [w1, w2, w3]

출력 C' = w1·B + w2·C + w3·D
`),o(h,null,{default:i(()=>[...m[0]||=[a(`ul`,null,[a(`li`,null,[e(`압축된 Q/K에 `),a(`strong`,null,`local context`),e(` 부여 후 attention score 계산`)]),a(`li`,null,[a(`strong`,null,`V에는 미적용`),e(` — content 자체를 담당, attention score와 무관`)]),a(`li`,null,[e(`동일 압축률에서 `),a(`strong`,null,`CCA > MLA`),e(` 성능 (CCA paper, Oct 2025)`)])],-1)]]),_:1})]),_:1},16)}}};export{m as default};