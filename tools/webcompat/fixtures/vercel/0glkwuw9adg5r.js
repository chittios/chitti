;!function(){try { var e="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof global?global:"undefined"!=typeof window?window:"undefined"!=typeof self?self:{},n=(new e.Error).stack;n&&((e._debugIds|| (e._debugIds={}))[n]="a3c0b598-2b04-ce7f-9a50-f060ccf59e7f")}catch(e){}}();
(globalThis["TURBOPACK_remote_chunk_loading_global_vercel-marketing"]||(globalThis["TURBOPACK_remote_chunk_loading_global_vercel-marketing"]=[])).push(["object"==typeof document?document.currentScript:void 0,9092058,e=>{"use strict";var t=e.i(7150081),n=e.i(6723374),a=e.i(4485205);function o({values:e}){return a.default.createElement("script",{type:"application/json","data-flag-values":!0,dangerouslySetInnerHTML:{__html:JSON.stringify(e,void 0,void 0).replace(/</g,"\\u003c")}})}e.s(["FlagValuesClient",0,function({flagValues:e}){return(0,n.useServerInsertedHTML)(()=>(0,t.jsx)(o,{values:e})),null}],9092058)},6275889,8612999,e=>{"use strict";var t=e.i(4485205),n=e.i(6723374);function a(e,t){if(!e||!t)return e;let n=e;for(let[e,a]of Object.entries(t)){if("lang"===e)continue;let t=Array.isArray(a)?a.join("/"):a??"";if(!t)continue;let o=t.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),r=RegExp(`/${o}(?=/|\\?|#|$)`),i=`/[${Array.isArray(a)?`...${e}`:e}]`;n=n.replace(r,i)}return n}e.s(["computeRoute",0,a],8612999),e.s(["useCurrentRoute",0,function(){let e=(0,n.usePathname)(),o=(0,n.useParams)();return(0,t.useMemo)(()=>{if(!o)return a(e,{});let{catchAll:t,...n}=o;return a(e,n)},[e,o])}],6275889)},8939773,e=>{"use strict";var t=e.i(4691250),n=e.i(7150081),a=e.i(4485205),o=e.i(4597921),r=e.i(9714397),i=e.i(6275889),s=e.i(6723374),u=e.i(7194621),l=e.i(5773208),c=e.i(2456388);let _="suspense_fallback_to_content_time",d=(0,r.metric)(_,{version:1}),S=(0,r.metric)("suspense_bad_fallback_shown",{version:1}),E="true"===t.default.env.NEXT_PUBLIC_DEBUG_NAVIGATION_METRICS;function f(){return!0===(0,o.getClientFlagValue)("enable-swr-hooks-in-migration")?"enabled":"disabled"}function m(e,t,n){let a=JSON.stringify(e,t,n);return"string"==typeof a?a.replace(/</g,"\\u003c"):a}function p(){return"production"}function k(e){return"production"===e}function A(e){return"1"===t.default.env.NEXT_PUBLIC_LOG_SUSPENSE_METRICS&&("development"===e||"preview"===e)}function w(e){let t="fallback"===e.status&&void 0!==e.contentCommittedAt?void 0!==e.fallbackShownAt?"resolved":"content":e.status;return t===e.status?e:{...e,status:t}}function b(){if(E)return window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__||(window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__=new Map),window.__VERCEL_GET_TRACKED_SUSPENSE_BOUNDARIES__=()=>Array.from(window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__?.values()??[]).map(w).sort((e,t)=>(e.mountedAt??e.fallbackShownAt??e.contentCommittedAt??0)-(t.mountedAt??t.fallbackShownAt??t.contentCommittedAt??0)),window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__}function R({boundaryId:e,name:t,route:n,hasExplicitName:a,status:o,mountedAt:r,fallbackShownAt:i,contentCommittedAt:s,navigationContext:u,badFallback:l}){let c=b();if(!c)return;let _=c.get(e),d=i??_?.fallbackShownAt,S=s??_?.contentCommittedAt,E=void 0!==n&&"unknown"!==n?n:_?.route!==void 0&&"unknown"!==_.route?_.route:"unknown",f=void 0!==d&&void 0!==S?S-d:_?.fallbackToContentMs;c.set(e,{boundaryId:e,name:t??_?.name,route:E,hasExplicitName:a??_?.hasExplicitName??void 0!==t,status:"fallback"===o&&void 0!==S?void 0!==d?"resolved":"content":o,mountedAt:r??_?.mountedAt??performance.now(),fallbackShownAt:d,contentCommittedAt:S,fallbackToContentMs:f,navigationContext:u??_?.navigationContext,badFallback:l??_?.badFallback})}function C(){return window.__VERCEL_SUSPENSE_FALLBACK_COMMITS__||(window.__VERCEL_SUSPENSE_FALLBACK_COMMITS__=new Map),window.__VERCEL_SUSPENSE_FALLBACK_COMMITS__}function h(){return window.__VERCEL_ROUTER_CONTENT_RENDER_PENDING_TRANSITION__?"router_transition":window.__VERCEL_ROUTER_TRANSITION_HAS_STARTED__?"non_router_transition":"initial_load"}function T({boundaryId:e,name:t,hasExplicitName:o=void 0!==t,badFallback:r,children:i}){(0,a.useLayoutEffect)(()=>{!function(e,{name:t,hasExplicitName:n,badFallback:a,routePathname:o}){let r=C();if(!r||r.has(e))return;let i=h(),s=b()?.get(e);if(s?.contentCommittedAt!==void 0)return R({boundaryId:e,name:t,route:o,hasExplicitName:n,status:void 0!==s.fallbackShownAt?"resolved":"content",navigationContext:i,badFallback:a});let l=performance.now();r.set(e,{routePathname:o,renderedAt:l,navigationContext:i}),R({boundaryId:e,name:t,route:o,hasExplicitName:n,status:"fallback",mountedAt:l,fallbackShownAt:l,navigationContext:i,badFallback:a}),(0,u.onSuspenseBoundaryFallbackWithRenderTiming)(e)}(e,{name:t,hasExplicitName:o,badFallback:r,routePathname:"unknown"})},[r,e,o,t]);let l=m(e),c=m("unknown"),_=m(t??null),d=m(o),S=m(!!r),f=E?`
  const trackedSuspenseBoundaries = (window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__ ||= new Map());
  window.__VERCEL_GET_TRACKED_SUSPENSE_BOUNDARIES__ ||= () =>
    Array.from((window.__VERCEL_TRACKED_SUSPENSE_BOUNDARIES__ || new Map()).values()).sort(
      (a, b) =>
        (a.mountedAt ?? a.fallbackShownAt ?? a.contentCommittedAt ?? 0) -
        (b.mountedAt ?? b.fallbackShownAt ?? b.contentCommittedAt ?? 0),
    );
  const previousTrackedSuspenseBoundary =
    trackedSuspenseBoundaries.get(${l});
  const trackedSuspenseFallbackShownAt =
    previousTrackedSuspenseBoundary?.fallbackShownAt ?? performance.now();
  const trackedSuspenseStatus =
    previousTrackedSuspenseBoundary?.contentCommittedAt !== undefined
      ? previousTrackedSuspenseBoundary.fallbackShownAt !== undefined
        ? 'resolved'
        : 'content'
      : 'fallback';
  trackedSuspenseBoundaries.set(${l}, {
    ...previousTrackedSuspenseBoundary,
    boundaryId: ${l},
    name: ${_} ?? undefined,
    route:
      previousTrackedSuspenseBoundary?.route !== undefined &&
      previousTrackedSuspenseBoundary.route !== 'unknown'
        ? previousTrackedSuspenseBoundary.route
        : fallbackRoutePathname,
    hasExplicitName: ${d},
    status: trackedSuspenseStatus,
    mountedAt:
      previousTrackedSuspenseBoundary?.mountedAt ??
      trackedSuspenseFallbackShownAt,
    fallbackShownAt: trackedSuspenseFallbackShownAt,
    navigationContext: fallbackNavigationContext,
    badFallback: ${S},
  });
`:"",p=`
{
  const fallbackCommits = (window.__VERCEL_SUSPENSE_FALLBACK_COMMITS__ ||= new Map());
  const fallbackRoutePathname = ${c};
  const fallbackNavigationContext =
    window.__VERCEL_ROUTER_CONTENT_RENDER_PENDING_TRANSITION__
      ? 'router_transition'
      : window.__VERCEL_ROUTER_TRANSITION_HAS_STARTED__
        ? 'non_router_transition'
        : 'initial_load';
  if (!fallbackCommits.has(${l})) {
    fallbackCommits.set(${l}, {
      renderedAt: performance.now(),
      routePathname: fallbackRoutePathname,
      navigationContext: fallbackNavigationContext,
    });
    (window.__VERCEL_SUSPENSE_RENDER_TIMING_PENDING__ ||= []).push(${l});
  }
${f}
}
`;return(0,s.useServerInsertedHTML)(()=>(0,n.jsx)("script",{dangerouslySetInnerHTML:{__html:p}})),(0,n.jsx)(n.Fragment,{children:i})}function v({boundaryId:e,name:t,hasExplicitName:o=void 0!==t,pleaseRemoveMeAndUseInstantInsights:r=!1,reportMetric:s=!0,children:l}){let c=(0,i.useCurrentRoute)()??"unknown",S=(0,a.useRef)(c),E=(0,a.useRef)(!1);return(0,a.useLayoutEffect)(()=>{if(!function(){let e=window.__VERCEL_SUSPENSE_RENDER_TIMING_PENDING__;if(e&&0!==e.length)for(let t of e.splice(0))(0,u.onSuspenseBoundaryFallbackWithRenderTiming)(t)}(),E.current)return;(0,u.onSuspenseBoundaryContentWithRenderTiming)(e);let n=performance.now(),a=C(),i=a?.get(e),l=i?.routePathname!==void 0&&"unknown"!==i.routePathname?i.routePathname:S.current;if(R({boundaryId:e,name:t,route:l,hasExplicitName:o,status:i?"resolved":"content",contentCommittedAt:n,navigationContext:i?.navigationContext,badFallback:r}),!i){E.current=!0;return}let c=n-i.renderedAt,m=p(),w={name:t??l,route_pathname:l,navigation_context:i.navigationContext??"initial_load",is_re_suspension:E.current,target_env:m,bad_fallback:r,swr_hooks_migration:f()};s&&k(m)&&d(c,w),s&&A(m)&&console.log(_,{value:c,attributes:w}),E.current=!0,a?.delete(e)},[e,o,t,r,s]),(0,a.useEffect)(()=>()=>{let t,n;C()?.delete(e),t=b(),n=t?.get(e),t&&n&&t.set(e,{...n,status:"unmounted"}),(0,u.onSuspenseBoundaryUnmountWithRenderTiming)(e)},[e]),(0,n.jsx)(n.Fragment,{children:l})}function g(){return(0,a.useEffect)(()=>{let e=p(),t={route:"unknown",from_route:l.routerState.lastRouterTransition.fromRoute,navigation_context:h(),target_env:e,swr_hooks_migration:f()};try{let e=l.routerState.lastRouterTransition.startMsSinceOrigin,n=performance.now();performance.measure("bad fallback shown",{start:0!==e?e:n,end:n,detail:{devtools:{track:"Bad Fallbacks ▲",properties:Object.entries(t).map(([e,t])=>[(0,c.snakeCaseToTitleCase)(e),String(t)])}}})}catch{}k(e)?S(1,t):A(e)&&console.log("suspense_bad_fallback_shown",{value:1,attributes:t})},[]),null}e.s(["InstantInsightsPendingSuspense",0,function({fallback:e,children:t}){let o=(0,a.useId)(),r=`suspense-instant-insights-${o}`;return(0,n.jsx)(a.Suspense,{fallback:(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(T,{boundaryId:r,badFallback:!0,children:e}),(0,n.jsx)(g,{})]}),children:(0,n.jsx)(v,{boundaryId:r,pleaseRemoveMeAndUseInstantInsights:!0,children:t})})},"NullFallback",0,g,"TrackedSuspenseContentMarker",0,v,"TrackedSuspenseDebugBoundary",0,function({name:e,fallback:t,children:o}){let r=(0,a.useId)(),i=`suspense-debug-${r}`,s=void 0!==e;return(0,n.jsx)(a.Suspense,{fallback:(0,n.jsx)(T,{boundaryId:i,name:e,hasExplicitName:s,children:t}),children:(0,n.jsx)(v,{boundaryId:i,name:e,hasExplicitName:s,reportMetric:!1,children:o})})},"TrackedSuspenseFallbackMarker",0,T])},2456388,e=>{"use strict";let t={hard:{skeleton:"primary-light",content:"primary-dark"},soft:{skeleton:"secondary-light",content:"secondary-dark"}};function n(e){return e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ")}e.s(["getDevtoolsDetail",0,function(e,a){let o=e?"hard":"soft";return{track:"Navigation Phases ▲",color:t[o][a.phase],properties:[["Navigation Type",o]].concat(Object.entries(a).map(([e,t])=>[n(e),String(t)]))}},"snakeCaseToTitleCase",0,n],2456388)}]);

//# debugId=a3c0b598-2b04-ce7f-9a50-f060ccf59e7f
//# sourceMappingURL=2444buuqwqbzs.js.map